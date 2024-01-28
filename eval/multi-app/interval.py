import numpy as np


def calculate_intervals(premium_comp, premium_comm, victim_comp, victim_comm, func):
    time_limit = func(premium_comp, premium_comm, victim_comp, victim_comm)
    premium = [[0, premium_comm]]
    time = premium_comm + premium_comp
    while time < time_limit:
        premium.append([time, time + premium_comm])
        time += premium_comm + premium_comp
    print("premium:", (np.array(premium) * 1000).tolist())

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
    print("victim:", (np.array(victim) * 1000).tolist())
    print("time_limit:", time_limit * 1000)


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

gpt_comp = 930
gpt_comm = 115
resnet_comp = 230
resnet_comm = 30

calculate_intervals(
    resnet_comp,
    resnet_comm,
    gpt_comp,
    gpt_comm,
    my_lcm,
)