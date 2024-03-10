import os
import glob
import csv
import random


def get_spk_to_utterance(data_dir) -> dict:
    """Get the dict of speaker to list of utterance for librispeech"""
    audio_files = glob.glob(os.path.join(data_dir, "*", "*", "*.flac"))
    spk_to_utts = dict()
    for file in audio_files:
        basename = os.path.basename(file)
        spk_id = basename.split('-')[0]
        if spk_id not in spk_to_utts:
            spk_to_utts[spk_id] = [file]
        else:
            spk_to_utts[spk_id].append(file)

    return spk_to_utts


def get_spk_to_utterance_csv(file) -> dict:
    """Get the dict of speaker to list of utterance from csv"""
    spk_to_utts = dict()
    with open(file) as f:
        reader = csv.reader(f)
        for line in reader:
            if len(line) != 2:
                continue
            spk_id = line[0].strip()
            filename = line[1].strip()

            if spk_id not in spk_to_utts:
                spk_to_utts[spk_id] = [filename]
            else:
                spk_to_utts[spk_id].append(filename)
    return spk_to_utts


def get_triplet(spk_to_utts):
    """Get a triplet of anchor/pos/neg samples."""
    pos_spk, neg_spk = random.sample(list(spk_to_utts.keys()), 2)
    # Retry if too few positive utterances.
    while len(spk_to_utts[pos_spk]) < 2:
        pos_spk, neg_spk = random.sample(list(spk_to_utts.keys()), 2)
    anchor_utt, pos_utt = random.sample(spk_to_utts[pos_spk], 2)
    neg_utt = random.sample(spk_to_utts[neg_spk], 1)[0]

    return anchor_utt, pos_utt, neg_utt




