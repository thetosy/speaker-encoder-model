import os
import torch
import multiprocessing

# train speech data path
TRAIN_DATA_DIR = os.path.join(
    os.path.expanduser("~"),
    "speech-data/LibriSpeech/dev-clean")

# test data path
TEST_DATA_DIR = os.path.join(os.path.expanduser("~"), "speech-data/LibriSpeech/test-clean")

# if given train and test dir is ignored
TRAIN_DATA_CSV = ""
TEST_DATA_CSV = ""

# saved model path
SAVED_MODEL_PATH = os.path.join(
    os.path.expanduser("~"),
    "speech-data/models/saved_model/saved_model.pt")

# Number of MFCCs
N_MFCC = 40

# Sequence length of the sliding window.
SEQ_LEN = 100  # 3.2 seconds

# Sliding window step for sliding window inference.
SLIDING_WINDOW_STEP = 50  # 1.6 seconds

# Dimension of transformer layers.
TRANSFORMER_DIM = 32

# Number of encoder layers for transformer
TRANSFORMER_ENCODER_LAYERS = 2

# Number of heads in transformer layers.
TRANSFORMER_HEADS = 8

# Alpha for the triplet loss.
TRIPLET_ALPHA = 0.1

# How many triplets do we train in a single batch.
BATCH_SIZE = 8

# Learning rate.
LEARNING_RATE = 0.0001

# Save a model to disk every these many steps.
SAVE_MODEL_FREQUENCY = 10000

# Number of steps to train.
TRAINING_STEPS = 100000

# Whether we are going to train with SpecAugment.
SPECAUG_TRAINING = True

# Parameters for SpecAugment training.
FREQ_MASK_PROB = 0.3
TIME_MASK_PROB = 0.3
FREQ_MASK_MAX_WIDTH = N_MFCC // 5
TIME_MASK_MAX_WIDTH = SEQ_LEN // 5

# Whether to use full sequence inference or sliding window inference.
USE_FULL_SEQUENCE_INFERENCE = False

# Number of triplets to evaluate for computing Equal Error Rate (EER).
# Both the number of positive trials and number of negative trials will be
# equal to this number.
NUM_EVAL_TRIPLETS = 10000

# Step of threshold sweeping for computing Equal Error Rate (EER).
EVAL_THRESHOLD_STEP = 0.001

# Number of processes for multi-processing.
NUM_PROCESSES = min(multiprocessing.cpu_count(), BATCH_SIZE)

# GPU or CPU.
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#if __name__ == "__main__":
    #print(DEVICE)


