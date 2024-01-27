#!/usr/bin/env bash

WORKDIR=`dirname $(realpath $0)`

usage() {
	echo "Usage: $0 <ring_type>     ring_type=goodring|badring"
}

if [ $# -ne 1 ]; then
	usage
	exit 1
fi

ring_type=$1

$WORKDIR/run_nccl_job_blue.sh $ring_type 2>&1 > /tmp/nccl_setting2_blue.stdout &
$WORKDIR/run_nccl_job_small.sh green  2>&1 > /tmp/nccl_setting2_green.stdout &
$WORKDIR/run_nccl_job_small.sh red 2>&1 > /tmp/nccl_setting2_red.stdout &

tail -f /tmp/nccl_setting2_blue.stdout /tmp/nccl_setting2_green.stdout /tmp/nccl_setting2_red.stdout
