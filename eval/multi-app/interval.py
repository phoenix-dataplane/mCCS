premium_comp = 310
premium_comm = 110
time_limit = (premium_comp + premium_comm) * 10
premium = [(0, premium_comm)]
time = premium_comm + premium_comp
while time < time_limit:
    premium.append((time, time + premium_comm))
    time += premium_comm + premium_comp
print("premium:", premium)


def inverse_intervals(intervals: list, time_limit: int):
    intervals = sorted(intervals, key=lambda x: x[1])
    intervals = intervals
    intervals.append((time_limit, time_limit))
    return [(intervals[i][1], intervals[i + 1][0]) for i in range(len(intervals) - 1)]


inverse_premium = inverse_intervals(premium, time_limit)


# first fit
victim_comp = 230
victim_comm = 35
victim = []
time = 0
for start, end in inverse_premium:
    time = max(time, start)
    while time + victim_comm < end:
        victim.append((time, time + victim_comm))
        time += victim_comm + victim_comp
print("victim:", victim)
