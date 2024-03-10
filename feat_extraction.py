import functools

import librosa
import torch
import random
import numpy as np
import config
import dataset
import specaug


def extract_audio_features(audio_file):
    """Extract MFCC from audio file. Return features in shape (time, mfcc)"""
    waveform, sr = librosa.load(audio_file, sr=None)

    if waveform.shape == 2:
        # convert to 1 channel
        waveform = librosa.to_mono(waveform)

    if sr != 16000:
        waveform = librosa.resample(waveform, sr, 16000)
        sr = 16000

    features = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=config.N_MFCC)
    return features.T


def get_triplet_features(spk_to_utts) -> tuple:
    """Get a triplet of anchor/pos/neg features."""
    anc, pos, neg = dataset.get_triplet(spk_to_utts)
    return extract_audio_features(anc), extract_audio_features(pos), extract_audio_features(neg)


def get_sliding_window_features(features) -> list:
    """Extract sliding window features"""
    start = 0
    sliding_window = []
    while start + config.SEQ_LEN < len(features.shape[0]):
        sliding_window.append(features[start: start + config.SEQ_LEN, :])
        start += config.SLIDING_WINDOW_STEP

    return sliding_window


def trim_features(features, apply_sec_agu):
    """get features to length SEQ_LEN"""
    start = random.randint(0, features.shape[0] - config.SEQ_LEN)
    trim_feat = features[start: start + config.SEQ_LEN, :]
    if apply_sec_agu:
        trim_feat = specaug.spec_aug(trim_feat)
    return trim_feat


def get_trimmed_triplet_features(_, spk_to_utts):
    """Get a triplet of trimmed anchor/pos/neg features."""
    anc, pos, neg = get_triplet_features(spk_to_utts)
    while anc.shape[0] < config.SEQ_LEN or pos.shape[0] < config.SEQ_LEN or neg.shape[0] < config.SEQ_LEN:
        anc, pos, neg = get_triplet_features(spk_to_utts)
        # create 3 d array of shape (3, SEQ_LEN, N_MFCC)
    return np.stack([trim_features(anc, config.SPECAUG_TRAINING),
                     trim_features(pos, config.SPECAUG_TRAINING),
                     trim_features(neg, config.SPECAUG_TRAINING)])


def get_batch_triplet_input(spk_to_utts, batch_size, pool=None):
    """Get batch triplet input. Returns Pytorch Tensor"""
    fetcher = functools.partial(get_trimmed_triplet_features, spk_to_utts=spk_to_utts)
    if pool is None:
        input_array = list(map(fetcher, range(batch_size)))
    else:
        # use multiprocessing
        input_array = pool.map(fetcher, range(batch_size))

    return torch.from_numpy(np.concatenate(input_array)).float()
