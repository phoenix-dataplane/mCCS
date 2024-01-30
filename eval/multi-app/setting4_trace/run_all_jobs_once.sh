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

OUTPUT_DIR=/tmp/nccl_setting4_trace

for job in {vgg,gpt_1,gpt_2}; do
	echo $job
	$WORKDIR/run_nccl_job.sh $job |& tee $OUTPUT_DIR/$trial_id.$job.stdout &
done

wait
