#!/usr/bin/env python3

# Creates a new EmoV-DB dataset by combining existing files and
# trimming/ignoring instructions from a CSV.

import glob
import os
import re
import csv
import pathlib
import shutil
import math


trim_instructions = {}
with open("/temp/e-liang/Trim - derp.csv", 'r', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        trim_instructions[row['File'][len("https://random.xgpf0.eliang.work"):]] = row

files = sorted(glob.glob(f"/temp/e-liang/out-no-silence/*/*/*.wav"))
for file in files:
    target_file = re.sub('out-no-silence', 'out-no-nve', file, 1)

    # Create containing folder
    target_folder = os.path.dirname(target_file)
    pathlib.Path(target_folder).mkdir(parents=True, exist_ok=True)

    if file in trim_instructions:
        instructions = trim_instructions[file]

        if len(instructions["NVE"]) > 0:
            continue # Audio contains untrimmable non-verbal expression; ignore file
        
        start_string = instructions["Start"]
        end_string = instructions["End"]

        # Don't trim if we don't need to
        if len(start_string) > 0 or len(end_string) > 0:
            def float_to_time_string(float_time):
                return f"0:{float(float_time)}"

            command = f"sox {file} {target_file} trim " \
                + (f"{float_to_time_string(start_string)} " if len(start_string) > 0 else "0 ") \
                + (f"={float_to_time_string(end_string)} " if len(end_string) > 0 else "") \

            print("Following instructions", instructions, command)
            os.system(command)

            continue # Since we trimmed, we're not passing this file through.

    print("Passthrough", file)
    shutil.copyfile(file, target_file)