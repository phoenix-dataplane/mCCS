#!/usr/bin/env bash

WORKDIR=`dirname $(realpath $0)`

usage() {
	echo "Usage: $0 <ring_type> <trial_id>    ring_type=goodring|badring"
}

if [ $# -ne 2 ]; then
	usage
	exit 1
fi

ring_type=$1
trial_id=$2

OUTPUT_DIR=/tmp/nccl_setting3

$WORKDIR/run_nccl_job_small.sh $ring_type blue |& tee $OUTPUT_DIR/$trial_id.blue.stdout &
$WORKDIR/run_nccl_job_small.sh $ring_type red |& tee $OUTPUT_DIR/$trial_id.red.stdout &

wait
