import numpy as np
import random
import config


def spec_aug(feat):
    """apply SpecAugment to features"""
    seq_len, n_mfcc = feat.shape
    outputs = feat
    mean_feat = np.mean(feat)

    if random.random() < config.FREQ_MASK_PROB:
        width = random.randint(1, config.FREQ_MASK_MAX_WIDTH)
        start = random.randint(0, n_mfcc - width)
        outputs[:, start: start + width] = mean_feat

    if random.random() < config.TIME_MASK_PROB:
        width = random.randint(1, config.TIME_MASK_MAX_WIDTH)
        start = random.randint(0, seq_len - width)
        outputs[start: start + width, :] = mean_feat

    return outputs

