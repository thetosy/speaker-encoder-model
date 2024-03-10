import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import multiprocessing

import dataset
import feat_extraction
import config


class TransformerEncoder(nn.Module):
    def __init__(self, saved_model=""):
        super(TransformerEncoder, self).__init__()
        # transformer network
        self.linear = nn.Linear(config.N_MFCC, config.TRANSFORMER_DIM)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=config.TRANSFORMER_DIM,
            nhead=config.TRANSFORMER_HEADS,
            batch_first=True), num_layers=config.TRANSFORMER_ENCODER_LAYERS)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d_model=config.TRANSFORMER_DIM,
            nhead=config.TRANSFORMER_HEADS,
            batch_first=True), num_layers=1)

        if saved_model:
            self._load(saved_model)

    def forward(self, x):
        encoder_input = torch.sigmoid(self.linear(x))
        encoder_output = self.encoder(encoder_input)
        tgt = torch.zeros(x.shape[0], 1, config.TRANSFORMER_DIM).to(
            config.DEVICE)
        output = self.decoder(tgt, encoder_output)
        return output[:, 0, :]

    def _load(self, saved_model):
        var_dict = torch.load(saved_model, map_location=config.DEVICE)
        self.load_state_dict(var_dict["encoder_state_dict"])


def get_encoder(load_from=""):
    """Create speaker encoder model or load it from a saved model."""
    return TransformerEncoder(load_from).to(config.DEVICE)


def get_triplet_loss(anc, pos, neg):
    """Triplet loss defined in https://arxiv.org/pdf/1705.02304.pdf."""
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    return torch.maximum(cos(neg, anc) - cos(pos, anc) + config.TRIPLET_ALPHA, torch.tensor(0.0))


def get_triplet_loss_batch(batch_input, batch_size):
    """get triplet loss from batch"""
    batch_reshape = torch.reshape(batch_input, (batch_size, 3, batch_input.shape[1]))
    # [:, n, :] - get the n index row from every batch into one matrix of size (batch_size, cols)
    loss = get_triplet_loss(batch_reshape[:, 0, :],
                            batch_reshape[:, 1, :],
                            batch_reshape[:, 2, :])
    loss = torch.mean(loss)
    return loss


def save_model(save_model_path, encoder, losses, start_time):
    """save model"""
    training_time = time.time() - start_time
    os.makedirs(save_model_path, exist_ok=True)
    if not save_model_path.endswith(".pt"):
        save_model_path += ".pt"
    else:
        torch.save({"encoder_state_dict": encoder.state_dict(),
                    "losses": losses,
                    "training_time": training_time
                    }, save_model_path)


def train(spk_to_utts, num_steps, save_model_path=None, pool=None):
    """Train transformer encoder model"""
    start_time = time.time()
    losses = []
    model = get_encoder()

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    print("begin training")
    for step in range(num_steps):
        optimizer.zero_grad()
        batch_input = feat_extraction.get_batch_triplet_input(spk_to_utts, config.BATCH_SIZE, pool).to(config.DEVICE)

        # get batch output
        batch_output = model(batch_input)

        loss = get_triplet_loss_batch(batch_output, config.BATCH_SIZE)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print("step:", step, "/", num_steps, "loss:", loss.item())

        if save_model_path is not None and (step + 1 % config.SAVE_MODEL_FREQUENCY == 0):
            checkpoint = save_model_path
            if checkpoint.endswith(".pt"):
                checkpoint = checkpoint[:-3]
            checkpoint = checkpoint + ".ckpt-" + str(step + 1) + ".pt"
            save_model(checkpoint, model, losses, start_time)

    print("training end")
    if save_model_path is not None:
        save_model(save_model_path, model, losses, start_time)

    return losses


def run():
    if config.TRAIN_DATA_CSV:
        spk_to_utts = dataset.get_spk_to_utterance_csv(config.TRAIN_DATA_CSV)
        print("Training data:", config.TRAIN_DATA_CSV)
    else:
        spk_to_utts = dataset.get_spk_to_utterance(config.TRAIN_DATA_DIR)
        print("Training data: ", config.TRAIN_DATA_DIR)

    with multiprocessing.Pool(config.NUM_PROCESSES) as pool:
        losses = train(spk_to_utts, config.TRAINING_STEPS, config.SAVED_MODEL_PATH, pool)

    plt.plot(losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.show()


if __name__ == "__main__":
    run()
