#!/usr/bin/env bash

WORKDIR=`dirname $(realpath $0)`

usage() {
	echo "Usage: $0 <job>    job=blue|green|red"
}

if [ $# -ne 1 ]; then
	usage
	exit 1
fi

job=$1

case $job in
	blue)
		device_id="0,1"
		tclass=0
		num_channels=2
		cat > hostfile.$job <<EOF
danyang-02 slots=2
danyang-01 slots=2
EOF
		;;
	green)
		device_id=0
		tclass=106
		num_channels=2
		cat > hostfile.$job <<EOF
danyang-03 slots=1
danyang-05 slots=1
EOF
		;;
	red)
		device_id=1
		tclass=66
		num_channels=2
		cat > hostfile.$job <<EOF
danyang-03 slots=1
danyang-05 slots=1
EOF
		;;
	*)
		echo "Error: job should be either 'blue', 'green' or 'red', got $job"
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
	-x NCCL_MAX_NCHANNELS=$num_channels -x NCCL_MIN_NCHANNELS=$num_channels -x NCCL_IB_QPS_PER_CONNECTION=1 \
	-x NCCL_IB_TC=$tclass \
	-x NCCL_EPOCHS=20 \
		$WORKDIR/../build/all_reduce_perf --datatype=half -b 128M -e 128M
