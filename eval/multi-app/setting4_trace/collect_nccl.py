#!/usr/bin/env python3

import os
import re
import sys
import csv
import glob
import os.path

OUTPUT_DIR = "/tmp/nccl_setting4_trace"

writer = csv.writer(sys.stdout)
writer.writerow(['setting','job','jct'])

def work(job: str, nccl_output_path) -> None:
    with open(nccl_output_path, 'r') as fin:
        for line in fin:
            if 'Total time' in line:
                tokens = line.split()
                # [vgg] [Rank 0] Total time: [69316] ms
                jct_ms = tokens[5].strip('[').strip(']')
                writer.writerow(['nccl-ecmp', job, jct_ms])

for path in glob.glob(os.path.join(OUTPUT_DIR, '*')):
    tokens = path.split('.')
    job = tokens[-2]
    work(job, path)
