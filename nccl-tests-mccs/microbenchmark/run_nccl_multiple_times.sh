#!/usr/bin/env bash

usage() {
	echo "Usage: $0 <num_iters> <num_gpus> <ring_type> <app>     num_gpus=1|2, ring_type=goodring|badring, app=allgather|allreduce"
}

if [ $# -ne 4 ]; then
	usage
	exit 1
fi

num_iters=$1
if [ $num_iters -gt 20 ]; then
	echo "$num_iters too large"
	exit 1
fi

shift
num_gpus=$1
ring_type=$2
app=$3

output_dir=/tmp/nccl_single_app

for i in `seq 1 $num_iters`; do
	echo Case $i
	./run_nccl_once.sh $num_gpus $ring_type $app |& tee $output_dir/${i}_${app}_${ring_type}_${num_gpus}.stdout
done
