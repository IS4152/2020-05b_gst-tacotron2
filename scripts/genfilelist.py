#!/usr/bin/env python3

# Generates filelists/FILENAMEPREFIXfilelist.txt files that contain paths to training audio
# files with their transcripts.
# 
# By default, this script must be run from within the scripts folder, i.e.:
#
# $ cd <repo root>/scripts
# $ ./genfilelist.py

import glob
import math
import os
import random
import re
import csv

import numpy as np

#####
# Params

# FILENAMEPREFIX = 'i_am_a_screw_up_and_forgot_to_set_this' # DO NOT COMMIT THIS UNCOMMENTED
# FILENAMEPREFIX = 'emovdb'
# FILENAMEPREFIX = 'emovdbwithspeaker'
FILENAMEPREFIX = 'emovdbwithonespeakeroneemo'
# FILENAMEPREFIX = 'emovdbbutoneonly'
# FILENAMEPREFIX = 'emovdbwithljsbutless'
# FILENAMEPREFIX = 'emovdbwithoutamused'
# FILENAMEPREFIX = 'meld'
# FILENAMEPREFIX = 'iemocap'

INCLUDE_EMOVDB = True
INCLUDE_MELD = False
INCLUDE_IEMOCAP = False

# SHOULD_REMOVE_AMUSED = True

SPLITS = [(0.98, "train"), (0.02, "val")]

EMOVDB_CMUARCTIC_PATH = '/home/e/e-liang/is4152/tacotron2/cmuarctic.data' # Transcripts for emovdb dataset located in repo root
MELD_FOLDER = '/temp/e-liang/MELD.Raw'
IEMOCAP_FOLDER = '/temp/e-liang/IEMOCAP_full_release'
OUT_FOLDER = '../filelists/' # Folder located in repo root

#####

def get_emovdb_lines():
    dataLookup = {}
    with open(EMOVDB_CMUARCTIC_PATH, 'r') as f:
        dataLines = f.readlines()
        p = re.compile('^\( arctic_a0(\d{3}) "(.*)"')
        for line in dataLines:
            m = p.match(line)
            if m is not None:
                (num, phrase) = m.group(1, 2)
                dataLookup[num] = phrase
                if num == '066':
                    dataLookup['666'] = phrase # To accomodate a typo in the file jenie/aud_am/amused_57-84_0666.wav

    p = re.compile('.*_0(\d{3}).wav')
    all_emovdb_lines = []

    # Regular speaker IDs
    # speakers = [os.path.basename(f) for f in glob.glob('/temp/e-liang/out/*')]
    # for speaker_idx, speaker in enumerate(speakers):
    #     files = sorted(glob.glob(f"/temp/e-liang/out/{speaker}/*/*"))
    #     all_emovdb_lines.extend([f"{os.path.abspath(file)}|{dataLookup[p.match(file).group(1)]}|{speaker_idx}\n" for file in files if p.match(file) is not None])

    # Select only 1 speaker for each of the emotions here.
    # Follows emo-tts:
    # Jenie anger, Jenie disgust (emo-tts uses Bea disgust, but Bea has no disgust emotion), Bea sleepiness, and Bea amused
    for speaker_idx, speaker in enumerate(['/jenie/aud_anger/', '/jenie/aud_disgust/', '/bea/amused/', '/bea/sleepiness/']):
        files = sorted(glob.glob(f"/temp/e-liang/out{speaker}*"))
        all_emovdb_lines.extend([f"{os.path.abspath(file)}|{dataLookup[p.match(file).group(1)]}|{speaker_idx}\n" for file in files if p.match(file) is not None])

    return all_emovdb_lines

def get_meld_lines():
    # Only use train split
    dataLookup = {}
    with open(os.path.join(MELD_FOLDER, 'train_sent_emo.csv'), 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleanedUtterance = row['Utterance'].encode('ascii', errors='ignore').decode()
            dataLookup[(row['Dialogue_ID'], row['Utterance_ID'])] = cleanedUtterance

    p = re.compile('dia(\d+)_utt(\d+)\.final\.wav$')
    meld_files = [(file, os.path.basename(file)) for file in glob.glob(os.path.join(MELD_FOLDER, '*', '*.final.wav'))]
    all_meld_lines = [f"{os.path.abspath(file)}|{dataLookup[(p.match(basename).group(1), p.match(basename).group(2))]}\n" for (file, basename) in meld_files if p.match(basename) is not None and (p.match(basename).group(1), p.match(basename).group(2)) in dataLookup]
    terminated_lines = [line for line in all_meld_lines if line.endswith('.\n')]
    return terminated_lines 

def get_iemocap_lines():
    dataLookup = {}
    p = re.compile('(\w+?) \[.+?\]: (.*)$')
    for file in glob.glob(os.path.join(IEMOCAP_FOLDER, 'Session*', 'dialog', 'transcriptions', '*.txt')):
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                match = p.match(line)
                if match is not None:
                    dataLookup[f"{match.group(1)}.wav"] = match.group(2)

    iemocap_files = [file for file in glob.glob(os.path.join(IEMOCAP_FOLDER, 'Session*', 'sentences', 'wav', '*', '*.wav'))]
    all_iemocap_lines = [f"{os.path.abspath(file)}|{dataLookup[os.path.basename(file)]}\n" for file in iemocap_files]
    return all_iemocap_lines

dataset_lines = []
if INCLUDE_EMOVDB:
    dataset_lines = get_emovdb_lines() + dataset_lines
if INCLUDE_MELD:
    dataset_lines = get_meld_lines() + dataset_lines
if INCLUDE_IEMOCAP:
    dataset_lines = get_iemocap_lines() + dataset_lines
random.shuffle(dataset_lines)

exit

splitLens = [math.floor(split * len(dataset_lines)) for (split, _) in SPLITS]
cumulativeSplitLens = np.cumsum(splitLens)

for (i, (_, name)) in enumerate(SPLITS):
    filesToExport = dataset_lines[slice(cumulativeSplitLens[i] - splitLens[i], cumulativeSplitLens[i])]
    filename = f"{FILENAMEPREFIX}_audio_text_{name}_filelist.txt"
    with open(os.path.join(OUT_FOLDER, filename), 'w+') as f:
        f.writelines(filesToExport)
