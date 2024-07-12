#!/usr/bin/env bash

WORKDIR=`dirname $(realpath $0)`

usage() {
	echo "Usage: $0 <trial_id>"
}

if [ $# -ne 1 ]; then
	usage
	exit 1
fi

trial_id=$1

OUTPUT_DIR=/tmp/nccl_setting1

$WORKDIR/run_nccl_job_small.sh blue |& tee $OUTPUT_DIR/$trial_id.blue.stdout &
$WORKDIR/run_nccl_job_small.sh red |& tee $OUTPUT_DIR/$trial_id.red.stdout &

wait
# tail -f /tmp/nccl_setting1_blue.stdout /tmp/nccl_setting1_red.stdout
