#!/usr/bin/env bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 [mccs|recover]"
    exit 1
fi

HOSTS=(danyang-01 danyang-05 danyang-02 danyang-03)

case $1 in
    mccs)
        cmd="echo -1 | sudo tee /sys/class/infiniband/mlx5_0/tc/1/traffic_class && sudo mlnx_qos -i rdma0 -r 0,0,47,47,0,0,0,0 --pfc 1,0,1,1,0,0,0,0 --tcbw=33,1,33,33,0,0,0,0 --tsa=ets,ets,ets,ets,ets,ets,ets,ets --prio2buffer=0,0,0,1,0,0,0,0"
        ;;
    recover)
        cmd="sudo /usr/local/bin/rdma_nic_config.sh && sudo mlnx_qos -i rdma0 -r 0,0,0,0,0,0,0,0"
        ;;
    *)
        echo "$0 [mccs|recover] get $1"
        exit 1
        ;;
esac

for h in ${HOSTS[@]}; do
    echo $h
    ssh -oStrictHostKeyChecking=no $h "$cmd"
done
