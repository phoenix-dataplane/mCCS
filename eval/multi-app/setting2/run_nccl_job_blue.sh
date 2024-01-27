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

case $ring_type in
	goodring)
		cat > hostfile.blue.$ring_type <<EOF
danyang-02 slots=1
danyang-03 slots=1
danyang-01 slots=1
danyang-05 slots=1
EOF
		;;
	badring)
		cat > hostfile.blue.$ring_type <<EOF
danyang-02 slots=1
danyang-01 slots=1
danyang-03 slots=1
danyang-05 slots=1
EOF
		;;
	*)
		echo "Error: ring_type should be either 'goodring' or 'badring', got $ring_type"
		usage
		exit 1
		;;
esac

mpirun --hostfile hostfile.blue.$ring_type -mca pml ob1 -mca btl tcp,self -mca btl_tcp_if_include eno1 \
	-x CUDA_VISIBLE_DEVICES=0 \
	-x NCCL_DEBUG=INFO -x NCCL_ALGO=Ring -x NCCL_PROTO=Simple \
	-x NCCL_IB_GID_INDEX=3 -x NCCL_SOCKET_IFNAME=rdma0 \
	-x NCCL_MAX_NCHANNELS=2 -x NCCL_MIN_NCHANNELS=2 -x NCCL_IB_QPS_PER_CONNECTION=1 \
	-x NCCL_IB_TC=106 \
	-x NCCL_EPOCHS=20 \
		$WORKDIR/../build/all_reduce_perf --datatype=half -b 128M -e 128M
