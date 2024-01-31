#!/usr/bin/env bash

usage() {
	echo "Usage: $0 <num_iters>"
}

if [ $# -ne 1 ]; then
	usage
	exit 1
fi

num_iters=$1
if [ $num_iters -gt 20 ]; then
	echo "$num_iters too large"
	exit 1
fi

suffix=`date +%Y%m%d.%H.%M.%S`
# output_dir=/tmp/${app}_${ring_type}_${num_gpus}gpus.${suffix}
output_dir=/tmp/nccl_single_app.${suffix}
mkdir -p $output_dir
unlink /tmp/nccl_single_app
ln -sf $output_dir /tmp/nccl_single_app

./run_nccl_multiple_times.sh $num_iters 1 badring allgather
./run_nccl_multiple_times.sh $num_iters 1 goodring allgather
./run_nccl_multiple_times.sh $num_iters 2 badring allgather
./run_nccl_multiple_times.sh $num_iters 2 goodring allgather

./run_nccl_multiple_times.sh $num_iters 1 badring allreduce
./run_nccl_multiple_times.sh $num_iters 1 goodring allreduce
./run_nccl_multiple_times.sh $num_iters 2 badring allreduce
./run_nccl_multiple_times.sh $num_iters 2 goodring allreduce
