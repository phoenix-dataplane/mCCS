#!/usr/bin/env python3

import os
import re
import sys
import csv
import glob
import os.path
import argparse

OUTPUT_DIR = "/tmp/nccl_single_app"

parser = argparse.ArgumentParser(description='')
parser.add_argument('--app', '--app', type=str,
                    help = 'The app of the trial, either allgather or allreduce')
parser.add_argument('--num-gpus', '--num-gpus', type=str,
                    help = 'The number of gpus to match, either 1 or 2')

args = parser.parse_args()
assert args.app
assert args.num_gpus

nccl_results = glob.glob(os.path.join(OUTPUT_DIR, "*{}_*_{}.stdout".format(args.app, args.num_gpus)))

writer = csv.writer(sys.stdout)
writer.writerow(['Solution', 'App', 'Size (Bytes)', 'Dtype', 'Latency (us)', 'AlgBW (GB/s)', 'BusBW (GB/s)'])

pat = re.compile(r'\s*\d+\s+\d+.*')
for path in nccl_results:
    with open(path, 'r') as fin:
        solution = path.split('/')[-1].split('_')[-2]
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
                writer.writerow([solution, args.app, size, dtype, latency_us, algbw, busbw])
