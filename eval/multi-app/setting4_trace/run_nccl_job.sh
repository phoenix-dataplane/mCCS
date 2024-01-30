#!/usr/bin/env bash

WORKDIR=`dirname $(realpath $0)`

usage() {
	echo "Usage: $0 <job>    job=vgg|gpt_1|gpt_2"
}

if [ $# -ne 1 ]; then
	usage
	exit 1
fi

job=$1

device_id="0,1"
case $job in
	vgg)
		tclass=0
		niters=101
		cat > hostfile.$job <<EOF
danyang-02 slots=2
danyang-01 slots=2
EOF
		;;
	gpt_1)
		tclass=106
		niters=1501
		cat > hostfile.$job <<EOF
danyang-03 slots=1
danyang-05 slots=1
EOF
		;;
	gpt_2)
		tclass=66
		niters=3001
		cat > hostfile.$job <<EOF
danyang-03 slots=1
danyang-05 slots=1
EOF
		;;
	*)
		echo "Error: job should be either 'vgg', 'gpt_1' or 'gpt_2', got $job"
		usage
		exit 1
		;;
esac

echo device_id = $device_id
echo Traffic class = $tclass

mpirun --hostfile hostfile.$job -mca pml ob1 -mca btl tcp,self -mca btl_tcp_if_include eno1 \
	-x CUDA_VISIBLE_DEVICES=$device_id \
	-x NCCL_DEBUG=INFO -x NCCL_ALGO=Ring -x NCCL_PROTO=Simple \
	-x NCCL_IB_GID_INDEX=3 -x NCCL_SOCKET_IFNAME=rdma0 \
	-x NCCL_MAX_NCHANNELS=2 -x NCCL_MIN_NCHANNELS=2 -x NCCL_IB_QPS_PER_CONNECTION=1 \
	-x NCCL_IB_TC=$tclass \
		$WORKDIR/../nccl-traffic-gen/nccl-traffic-gen -n $niters -j $job -p $WORKDIR/../nccl-traffic-gen/setup-4_$job.toml -v
