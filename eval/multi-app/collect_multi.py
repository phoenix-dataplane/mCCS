#!/usr/bin/env python3
import os


def convert_size(size: str):
    if size[-1] == "K":
        return int(size[:-1]) * 1024
    elif size[-1] == "M":
        return int(size[:-1]) * 1024 * 1024
    elif size[-1] == "G":
        return int(size[:-1]) * 1024 * 1024 * 1024
    else:
        return int(size)


def get_output(path: str):
    read_files = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.endswith(".stdout") and f.find("bench") != -1
    ]
    file_contents = []
    for f in read_files:
        with open(f, "r") as f:
            file_contents.append(f.read())
    return file_contents


def filter_contents(contents: list, filter_list: list):
    def filter(lines: str):
        lines = lines.split("\n")
        res = []
        for line in lines:
            contains_all = True
            for f in filter_list:
                if line.find(f) == -1:
                    contains_all = False
                    break
            if contains_all:
                res.append(line)
        return res

    res = []
    for i in [filter(content) for content in contents]:
        res.extend(i)
    return res


def collect_setup(base_dir: str, group: str, setup: int, app_cnt: int):
    path = os.path.join(base_dir, group, group + "-setup" + str(setup))
    outputs = get_output(path)
    res = []
    for i in range(1, app_cnt + 1):
        line = filter_contents(outputs, ["app" + str(i), "Rank 0", "Epoch=3"])
        res.append(
            (
                f"app{i}",
                line[0].split(": ")[-2].split(" GB")[0],
                line[0].split(": ")[-1].split(" GB")[0],
            )
        )
    return res


def collect_setup2(base_dir: str, group: str, each: str, setup: int, app_cnt: int):
    path = os.path.join(base_dir, group, each + "-setup" + str(setup))
    if not os.path.exists(path):
        return []
    outputs = get_output(path)
    res = []
    for i in range(1, app_cnt + 1):
        line = filter_contents(outputs, ["app" + str(i), "Rank 0", "Epoch=1"])
        res.append(
            (
                f"app{i}",
                line[0].split(": ")[-2].split(" GB")[0],
                line[0].split(": ")[-1].split(" GB")[0],
            )
        )
    return res


def interactive():
    if __name__ == "__main__":
        import sys

        # input: setup
        mapping = {1: 2, 2: 3, 3: 2}
        setup = int(sys.argv[1])
        collect_setup("/tmp", "multi-allreduce", setup, mapping[setup])


def collect_allreduce_all():
    mapping = {1: 2, 2: 3, 3: 2, 4: 3}
    app_mapping = {"app1": "blue", "app2": "red", "app3": "green"}
    for setup in [1, 2, 3, 4]:
        res = ""
        for i in range(20):
            for line in collect_setup2(
                "/tmp",
                f"multi-allreduce-ecmp-{i}",
                "multi-allreduce-ecmp",
                setup,
                mapping[setup],
            ):
                res += f"Multi-Allreduce-ECMP,{app_mapping[line[0]]},128M,float16,0,{line[1]},{line[2]},{i}\n"
        for i in range(20):
            for line in collect_setup2(
                "/tmp",
                f"multi-allreduce-flow-{i}",
                "multi-allreduce-flow",
                setup,
                mapping[setup],
            ):
                res += f"Multi-Allreduce-Flow,{app_mapping[line[0]]},128M,float16,0,{line[1]},{line[2]},{i}\n"
        with open(f"../plot/data/multi-setting-{setup}.csv", "w") as f:
            f.write(res)


if __name__ == "__main__":
    collect_allreduce_all()

# def reconfig_allreduce():
#     s = collect_setup2