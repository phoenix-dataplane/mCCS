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
    for i in range(1, app_cnt + 1):
        line = filter_contents(outputs, ["app" + str(i), "Rank 0", "Epoch=3"])
        print(f"app{i}", line[0].split(": ")[-1].split(" GB")[0])


if __name__ == "__main__":
    import sys

    # input: setup
    mapping = {1: 2, 2: 3, 3: 2}
    setup = int(sys.argv[1])
    collect_setup("/tmp", "multi-allreduce", setup, mapping[setup])
