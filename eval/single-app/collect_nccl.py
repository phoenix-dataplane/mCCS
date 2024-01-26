#!/usr/bin/env python3

import os
import re
import sys
import csv
import glob
import os.path
import argparse

OUTPUT_DIR = "/tmp/nccl_single_app"

parser = argparse.ArgumentParser(description='Launch a dsagent and deepscheduler')
parser.add_argument('--solution', '--solution', required=True, type=str,
                    help = 'Give a name to the trial, either NCCL Bad Ring|NCCL Good Ring.')
parser.add_argument('--app', '--app', type=str,
                    help = 'The app of the trial, either Allgather or Allreduce')

args = parser.parse_args()
assert args.solution

if args.app is not None:
    app = args.app
else:
    basename = os.readlink(OUTPUT_DIR).split('/')[-1]
    app = basename.split('_')[0]

nccl_results = glob.glob(os.path.join(OUTPUT_DIR, "*"))

writer = csv.writer(sys.stdout)
writer.writerow(['Solution', 'App', 'Size (Bytes)', 'Dtype', 'Latency (us)', 'AlgBW (GB/s)', 'BusBW (GB/s)'])

pat = re.compile(r'\s*\d+\s+\d+.*')
for path in nccl_results:
    with open(path, 'r') as fin:
        for line in fin:
            match = pat.match(line)
            if match is not None:
                # print(line.split())
                tokens = line.split()
                # ['536870912', '33554432', 'float', 'none', '-1', '115671', '4.64', '3.48', '0', '114558', '4.69', '3.51', '0']
                size = tokens[0]
                dtype = tokens[2]
                latency_us = tokens[9]
                algbw = tokens[10]
                busbw = tokens[11]
                writer.writerow([args.solution, app, size, dtype, latency_us, algbw, busbw])
