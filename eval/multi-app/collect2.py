# %%
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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
        if line.find("[Rank 0] Iter") != -1:
            prefix = line.split(" [Rank")[0][1:-1]
            iter_no = line.split("Iter ")[-1].split(" ")[0]
            time_ms = line.split("time: ")[-1].split(" ")[0]
            if prefix not in res:
                res[prefix] = []
            res[prefix].append((iter_no, time_ms))
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


get_cdf()

# %%
