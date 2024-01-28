#!/usr/bin/env python3

import os
import re
import sys
import csv
import glob
import os.path
import argparse

OUTPUT_DIR = "/tmp/nccl_setting3"

parser = argparse.ArgumentParser(description='Launch a dsagent and deepscheduler')
parser.add_argument('--solution', '--solution', required=True, type=str,
                    help = 'Give a name to the trial, either NCCL Bad Ring|NCCL Good Ring.')
parser.add_argument('--strip-head', '--strip-head', required=False, type=int, default=1,
                    help = 'Omit the first few lines of the output.')
parser.add_argument('--strip-tail', '--strip-tail', required=False, type=int, default=0,
                    help = 'Omit the last few lines of the output.')

args = parser.parse_args()
assert args.solution

writer = csv.writer(sys.stdout)
writer.writerow(['Solution', 'App', 'Size (Bytes)', 'Dtype', 'Latency (us)', 'AlgBW (GB/s)', 'BusBW (GB/s)', 'Trial ID'])

pat = re.compile(r'\s*\d+\s+\d+.*')

def get_latency(rec) -> int:
    return int(rec[4])

def get_job_duration(records) -> int:
    return sum(map(get_latency, records))

def work(app_color: str, trial_id, nccl_output_path) -> None:
    results = []
    with open(nccl_output_path, 'r') as fin:
        for line in fin:
            match = pat.match(line)
            if match is not None:
                tokens = line.split()
                # ['536870912', '33554432', 'float', 'none', '-1', '115671', '4.64', '3.48', '0', '114558', '4.69', '3.51', '0']
                size = tokens[0]
                dtype = tokens[2]
                latency_us = tokens[9]
                algbw = tokens[10]
                busbw = tokens[11]
                results.append([args.solution, app_color, size, dtype, latency_us, algbw, busbw, trial_id])
    return results

jobs = []

for path in glob.glob(os.path.join(OUTPUT_DIR, '*')):
    tokens = path.split('.')
    trial_id = tokens[-3].split('/')[-1]
    color = tokens[-2]
    jobs += [work(color, trial_id, path)]

# min_dura = min(map(get_job_duration, jobs))
# print(f'min job duration: {min_dura}')

for results in jobs:
    assert len(results) > args.strip_head + args.strip_tail, "len(results) = {}".format(len(results))
    if args.strip_tail == 0:
        stripped_results = results[args.strip_head:]
    else:
        stripped_results = results[args.strip_head:-args.strip_tail]
    for rec in stripped_results:
        writer.writerow(rec)
    # writer.writerow(rec)
