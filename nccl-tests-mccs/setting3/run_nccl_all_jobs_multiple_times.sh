#!/usr/bin/env bash

usage() {
	echo "Usage: $0 <num_iters> <ring_type>    ring_type=goodring|badring"
}

if [ $# -ne 2 ]; then
	usage
	exit 1
fi

num_iters=$1
if [ $num_iters -gt 20 ]; then
	echo "$num_iters too large"
	exit 1
fi

ring_type=$2

suffix=`date +%Y%m%d.%H.%M.%S`
output_dir=/tmp/nccl_setting3.${suffix}
mkdir -p $output_dir
unlink /tmp/nccl_setting3
ln -sf $output_dir /tmp/nccl_setting3


for i in `seq 1 $num_iters`; do
	echo Case $i
	./run_all_jobs_once.sh $ring_type $i
done
