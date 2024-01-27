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

OUTPUT_DIR=/tmp/nccl_setting2

$WORKDIR/run_nccl_job_blue.sh $ring_type |& tee $OUTPUT_DIR/$trial_id.blue.stdout &
$WORKDIR/run_nccl_job_small.sh green  |& tee $OUTPUT_DIR/$trial_id.green.stdout &
$WORKDIR/run_nccl_job_small.sh red |& tee $OUTPUT_DIR/$trial_id.red.stdout &

wait
# tail -f /tmp/nccl_setting2_blue.stdout /tmp/nccl_setting2_green.stdout /tmp/nccl_setting2_red.stdout
