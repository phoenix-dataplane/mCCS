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


def filter_contents(app: str, contents: list):
    def filter(lines: str):
        return [
            line
            for line in lines.split("\n")
            if line.find("bandwidth") != -1 and line.find(app) != -1
        ]

    res = []
    for i in [filter(content) for content in contents]:
        res.extend(i)
    return res


def rank0_parse_algo_bw(lines: list):
    for line in lines:
        if line.find("Rank 0") != -1:
            return line.split(": ")[-1].split(" GB")[0]
    print("Error: no rank 0 found: ", lines)


def collect_setup(base_dir: str, group: str, setup: int, app_cnt: int):
    path = os.path.join(base_dir, group, group + "-setup" + str(setup))
    outputs = get_output(path)
    for i in range(1, app_cnt + 1):
        # print(i, rank0_parse_bw(filter_contents("app" + str(i), outputs)).split(" ")[0])
        print(i, "\n".join(filter_contents("app" + str(i), outputs)))


if __name__ == "__main__":
    import sys

    # input: setup
    mapping = {1: 2, 2: 3, 3: 2}
    setup = int(sys.argv[1])
    collect_setup("/tmp", "multi-allreduce", setup, mapping[setup])
