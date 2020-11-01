#!/usr/bin/env python3

# Generates a list of files to be manually trimmed. To aid pre-processing of
# our existing files.

import glob
import os
import re

EMOVDB_CMUARCTIC_PATH = '/home/e/e-liang/is4152/tacotron2/cmuarctic.data' # Transcripts for emovdb dataset located in repo root

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
# for speaker_idx, speaker in enumerate(['/jenie/aud_anger/', '/jenie/aud_disgust/', '/bea/amused/', '/bea/sleepiness/']):
# We only want to manually trim files we intend to use
for speaker_idx, speaker in enumerate(['/bea/amused/', '/bea/sleepiness/']):
    files = sorted(glob.glob(f"/temp/e-liang/out-no-silence{speaker}*"))
    all_emovdb_lines.extend([f"https://random.xgpf0.eliang.work{file},,,\n" for file in files if p.match(file) is not None])

with open("/temp/e-liang/derp.csv", 'w+') as f:
    f.writelines(all_emovdb_lines)
