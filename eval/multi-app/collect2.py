# %%
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta


def file_paths(base_dir: str):
    res = []
    for group in os.listdir(base_dir):
        for each in os.listdir(os.path.join(base_dir, group)):
            if each.startswith("traffic_gen"):
                res.append(os.path.join(base_dir, group, each))
    return res


def parse_file(path: str):
    with open(path, "r") as f:
        lines = f.readlines()
    res = {}
    comp_time = []
    for line in lines:
        if line.find("Iter") != -1:
            prefix = line.split(" [Rank")[0][1:-1]
            iter_no = line.split("Iter ")[-1].split(" ")[0]
            time_ms = line.split("time: ")[-1].split(" ")[0]
            timestamp = line.split("<")[-1].split(">")[0]
            if prefix not in res:
                res[prefix] = []
            res[prefix].append((iter_no, time_ms, timestamp))
        if line.find("run time") != -1:
            prefix = line.split("Rank")[0][1:-1]
            time_ms = line.split("run time: ")[-1].split(" ")[0]
            comp_time.append((prefix, time_ms))
    return res, comp_time


def get_cdf():
    coll = {"vgg": {}, "gpt_1": {}, "gpt_2": {}}
    for i in file_paths("/tmp/setup4-cdf"):
        dic, _ = parse_file(i)
        setup = i.split("/")[-2].split("-")[-1]
        if len(dic) > 0:
            for k, v in dic.items():
                if k == "vgg":
                    start = 30
                    end = 200
                elif k == "gpt_1":
                    start = 100
                    end = 1000
                elif k == "gpt_2":
                    start = 100
                    end = 1000
                data = np.sort([float(i[1]) for i in v[start:end]])
                coll[k][setup] = data
    # plot CDF for each app
    for app, data in coll.items():
        plt.figure()
        data_min = min([min(i) for i in data.values()])
        data_max = max([max(i) for i in data.values()])
        # CDF for each category in data
        for k, v in data.items():
            y = np.arange(1, len(v) + 1) / len(v)
            plt.plot(v, y, label=k)
        # only show 3 x ticks
        plt.xticks(ticks=[data_min, (data_min + data_max) / 2, data_max])
        plt.xlabel("Time (ms)")
        plt.ylabel("CDF")
        plt.legend()
        plt.title(f"{app} CDF")


def moving_average(array, window_size):
    smoothed_array = []
    for i in range(len(array)):
        window = array[max(0, i - window_size + 1) : i + 1]
        smoothed_y = sum(point[1] for point in window) / len(window)
        smoothed_array.append((array[i][0], smoothed_y))
    return smoothed_array


def get_average(data):
    s = [i[1] for i in data if i[0] > 90000 and i[0] < 110000]
    print(sum(s) / len(s))


def get_trend_figure():
    vgg_path = "/tmp/dynamic-config/setup4-dynamic/traffic_gen_danyang-02.stdout"
    gpt1_path = "/tmp/dynamic-config/setup4-dynamic-gpt-1/traffic_gen_danyang-03.stdout"
    gpt2_path = "/tmp/dynamic-config/setup4-dynamic-gpt-2/traffic_gen_danyang-03.stdout"

    def convert_to_datetime(timestamp):
        dt = datetime.strptime(timestamp, "%H:%M:%S.%f")
        return (dt - datetime.min).total_seconds() * 1000

    baseline = {"vgg": 344.19, "gpt_1": 26.53, "gpt_2": 26.82}

    def extrect(data, tag):
        return [
            (convert_to_datetime(i[2]), 1 / (float(i[1]) / baseline[tag]))
            for i in data[tag]
        ]
        # return [(convert_to_datetime(i[2]), float(i[1])) for i in data[tag]]

    vgg, _ = parse_file(vgg_path)
    vgg = moving_average(extrect(vgg, "vgg"), 2)
    gpt1, _ = parse_file(gpt1_path)
    gpt1 = moving_average(extrect(gpt1, "gpt_1"), 30)
    gpt2, _ = parse_file(gpt2_path)
    gpt2 = moving_average(extrect(gpt2, "gpt_2"), 30)
    # (timestamp, throughput)
    # timestamp based on 0
    earliest_timestamp = min(min(vgg)[0], min(gpt1)[0], min(gpt2)[0])
    # moving average
    aligned_vgg = [
        (timestamp - earliest_timestamp, throughput) for timestamp, throughput in vgg
    ]
    aligned_gpt1 = [
        (timestamp - earliest_timestamp, throughput) for timestamp, throughput in gpt1
    ]
    aligned_gpt2 = [
        (timestamp - earliest_timestamp, throughput) for timestamp, throughput in gpt2
    ]
    aligned_vgg = [i for i in aligned_vgg if i[0] < 200000]
    aligned_gpt1 = [i for i in aligned_gpt1 if i[0] < 200000]
    aligned_gpt2 = [i for i in aligned_gpt2 if i[0] < 200000]
    plt.figure(figsize=(10, 6))
    plt.plot(
        [time for time, _ in aligned_vgg],
        [throughput for _, throughput in aligned_vgg],
        label="vgg",
    )
    plt.plot(
        [time for time, _ in aligned_gpt1],
        [throughput for _, throughput in aligned_gpt1],
        label="gpt1",
    )
    plt.plot(
        [time for time, _ in aligned_gpt2],
        [throughput for _, throughput in aligned_gpt2],
        label="gpt2",
    )
    plt.xlabel("Time")
    plt.ylabel("Throughput")
    plt.title("Throughput Over Time")
    plt.legend()
    plt.show()


get_trend_figure()
# get_cdf()


# %%
def get_reconfig_figure():
    gpt_path = "/tmp/ring-reconfig/setup4-dynamic/traffic_gen_danyang-02.stdout"

    def convert_to_datetime(timestamp):
        dt = datetime.strptime(timestamp, "%H:%M:%S.%f")
        return (dt - datetime.min).total_seconds() * 1000

    def extrect(data, tag):
        return [(convert_to_datetime(i[2]), 1 / (float(i[1])/1000)) for i in data[tag]]
        # return [(convert_to_datetime(i[2]), float(i[1])) for i in data[tag]]

    data = moving_average(extrect(parse_file(gpt_path)[0], "gpt"),3)
    earliest_timestamp = min(data)[0]
    aligned_vgg = [
        (timestamp - earliest_timestamp, throughput) for timestamp, throughput in data
    ]
   
    aligned_vgg = [i for i in aligned_vgg if i[0] > 24000 and i[0] < 35000]
    plt.figure(figsize=(10, 6))
    plt.plot(
        [time for time, _ in aligned_vgg],
        [throughput for _, throughput in aligned_vgg],
        label="GPT",
    )
   
    plt.xlabel("Time")
    plt.ylabel("Throughput")
    plt.title("Throughput Over Time")
    plt.legend()
    plt.show()

get_reconfig_figure()