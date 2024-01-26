#!/usr/bin/env bash

WORKDIR=`dirname $(realpath $0)`

usage() {
        echo "Usage: $0 <num_gpus> <ring_type>     num_gpus=1|2, ring_type=goodring|badring"
}

if [ $# -ne 2 ]; then
        usage
        exit 1
fi

num_gpus=$1
ring_type=$2

case $num_gpus in
        1)
                tclass=106
                ;;
        2)
                tclass=0
                ;;
        *)
                echo "Error: num_gpus should be either '1' or '2', got $num_gpus"
                usage
                exit 1
                ;;
esac

echo "Traffic class=$tclass"

case $ring_type in
        goodring)
                cat > hostfile.$ring_type <<EOF
danyang-02 slots=$num_gpus
danyang-03 slots=$num_gpus
danyang-01 slots=$num_gpus
danyang-05 slots=$num_gpus
EOF
                ;;
        badring)
                cat > hostfile.$ring_type <<EOF
danyang-02 slots=$num_gpus
danyang-01 slots=$num_gpus
danyang-03 slots=$num_gpus
danyang-05 slots=$num_gpus
EOF
                ;;
        *)
                echo "Error: ring_type should be either 'goodring' or 'badring', got $ring_type"
                usage
                exit 1
                ;;
esac

mpirun --hostfile hostfile.$ring_type -mca pml ob1 -mca btl tcp,self -mca btl_tcp_if_include eno1 \
        -x CUDA_VISIBLE_DEVICES=0,1 \
        -x NCCL_DEBUG=INFO -x NCCL_ALGO=Ring -x NCCL_PROTO=Simple \
        -x NCCL_IB_GID_INDEX=3 -x NCCL_SOCKET_IFNAME=rdma0 \
        -x NCCL_MAX_NCHANNELS=2 -x NCCL_MIN_NCHANNELS=2 \
        -x NCCL_IB_TC=$tclass \
                $WORKDIR/../build/all_gather_perf -b 32K -e 512M -f 4
