import numpy as np


def calculate_intervals(premium_comp, premium_comm, victim_comp, victim_comm, func):
    time_limit = func(premium_comp, premium_comm, victim_comp, victim_comm)
    premium = [[0, premium_comm]]
    time = premium_comm + premium_comp
    while time < time_limit:
        premium.append([time, time + premium_comm])
        time += premium_comm + premium_comp
    # print("premium:", (np.array(premium) * 1000).tolist())

    def inverse_intervals(intervals: list, time_limit: int):
        intervals = sorted(intervals, key=lambda x: x[1])
        intervals = intervals
        intervals.append((time_limit, time_limit))
        return [
            (intervals[i][1], intervals[i + 1][0]) for i in range(len(intervals) - 1)
        ]

    inverse_premium = inverse_intervals(premium, time_limit)

    # first fit
    victim = []
    time = 0
    for start, end in inverse_premium:
        time = max(time, start)
        while time + victim_comm < end:
            victim.append([time, time + victim_comm])
            time += victim_comm + victim_comp
    # print("victim:", (np.array(victim) * 1000).tolist())
    # print("time_limit:", time_limit * 1000)
    return premium, victim, time_limit


import math


def my_lcm(pre_comp, pre_comm, vic_comp, vic_comm):
    return math.lcm(pre_comp + pre_comm, vic_comp + vic_comm)


# vgg_comp = 310
# vgg_comm = 110
# resnet_comp = 230
# resnet_comm = 35

# calculate_intervals(
#     vgg_comp,
#     vgg_comm,
#     resnet_comp,
#     resnet_comm,
#     my_lcm,
# )

# gpt_comp = 930
# gpt_comm = 115
# resnet_comp = 230
# resnet_comm = 30

# calculate_intervals(
#     resnet_comp,
#     resnet_comm,
#     gpt_comp,
#     gpt_comm,
#     my_lcm,
# )


def fill_template(list1, list2, time, is_ecmp=False):
    if not is_ecmp:
        s_port1 = "udp_sport = [[1, 2, 49200], [3, 0, 49200]],"
        s_port2 = "udp_sport = [[0, 1, 49202], [1, 0, 49202]],"
    else:
        s_port1 = ""
        s_port2 = ""
    return (
        r"""
mccs_daemon_basename = "mccs-deamon"
mccs_daemon_prefix = "/tmp/mccs-${USER}"
addrs = [
    "0.0.0.0",
    "192.168.211.2",
    "192.168.211.34",
    "192.168.211.66",
    "192.168.211.130",
    "192.168.211.162",
    "192.168.211.195",
]
listen_port = 5000

[control]
prefix = "/tmp/mccs-${USER}"
path = "control.sock"

[comm_default_config]
buffer_sizes = [4194304]
channel_count = 2

[comm_global_config]
[comm_global_config.net_rdma]
gid_index = 3
qps_per_conn = 1
timeout = 18
retry_count = 7
pkey = 0
use_inline = false
service_level = 0
traffic_class = 0
adaptive_routing = false
ar_threshold = 8192
pci_relaxed_ordering = false
gdr_flush_disable = true
socket_if_prefix = "rdma"

[comm_global_config.net]
gdr_enable = false
gdr_copy_sync_enable = false
gdr_copy_flush_enable = false

[comm_global_config.shm]
locality = "Sender"
memcpy_send = false
memcpy_recv = false

[qos_schedule]
epoch_microsecs = """
        + str(time)
        + r""" 

[qos_schedule.schedule.201]
intervals = """
        + str(list1)
        + r"""
mode = "Allow"
        
[qos_schedule.schedule.202]
intervals = """
        + str(list2)
        + r"""

mode = "Deny"

# magic number: 49200 & 49202

[[comm_patterns_override]]
communicator_id = 200
channels = [
    { channel_id = 0, ring = [0, 1, 2, 3], """
        + s_port1
        + r""" net_dev = "mlx5_0" },
    { channel_id = 1, ring = [0, 1, 2, 3], """
        + s_port1
        + r""" net_dev = "mlx5_0" },
]
ib_traffic_class = 0

[[comm_patterns_override]]
communicator_id = 201
channels = [
    { channel_id = 0, ring = [0, 1],  """
        + s_port2
        + r""" net_dev = "mlx5_0" },
    { channel_id = 1, ring = [0, 1], """
        + s_port2
        + r""" net_dev = "mlx5_0" },
]
ib_traffic_class = 66


[[comm_patterns_override]]
communicator_id = 202
channels = [
    { channel_id = 0, ring = [0, 1],"""
        + s_port2
        + r""" net_dev = "mlx5_0" },
    { channel_id = 1, ring = [0, 1], """
        + s_port2
        + r""" net_dev = "mlx5_0" },
]
ib_traffic_class = 66
"""
    )


def gen_qos():
    gpt_comp = 930
    gpt_comm = 115
    vgg_comp = 310
    vgg_comm = 110
    gpt_comp = 6
    gpt_comm = 17
    intvl, victim, epoch_time = calculate_intervals(
        6,
        17,
        6,
        1,
        lambda x, y, z, w: 1000,
    )
    intvl = (np.array(intvl) * 1000).tolist()
    victim = (np.array(victim) * 1000).tolist()
    text = fill_template(intvl, intvl, epoch_time * 1000, False)
    with open("output/setup4-mccs-config.toml", "w") as f:
        f.write(text)
    text = fill_template(intvl, intvl, epoch_time * 1000, True)
    with open("output/setup4-mccs-ecmp-config.toml", "w") as f:
        f.write(text)


gen_qos()
