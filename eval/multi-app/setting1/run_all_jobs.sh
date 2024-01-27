#!/usr/bin/env bash

WORKDIR=`dirname $(realpath $0)`

$WORKDIR/run_nccl_job_small.sh blue  2>&1 > /tmp/nccl_setting1_blue.stdout &
$WORKDIR/run_nccl_job_small.sh red 2>&1 > /tmp/nccl_setting1_red.stdout &

tail -f /tmp/nccl_setting1_blue.stdout /tmp/nccl_setting1_red.stdout
