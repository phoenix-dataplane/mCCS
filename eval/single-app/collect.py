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


def get_output(base_dir: str, node_str: str, command: str, size: str):
    path = os.path.join(base_dir, node_str, command, size)
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


def filter_contents(contents: list):
    def filter(lines: str):
        return [line for line in lines.split("\n") if line.find("bandwidth") != -1]

    res = []
    for i in [filter(content) for content in contents]:
        res.extend(i)
    return res


def rank0_parse_bw(lines: list):
    for line in lines:
        if line.find("Rank 0") != -1:
            return line.split(": ")[-1]


def aggregrate(base_dir: str, node_str: str):
    ret = {}
    for command in os.listdir(os.path.join(base_dir, node_str)):
        val = {}
        for size in os.listdir(os.path.join(base_dir, node_str, command)):
            val[size] = rank0_parse_bw(
                filter_contents(get_output(base_dir, node_str, command, size))
            )
        # sort by size
        ret[command] = dict(sorted(val.items(), key=lambda item: convert_size(item[0])))
    return ret


if __name__ == "__main__":
    # read base dir and node_str from command line
    import sys

    if len(sys.argv) != 3:
        print("Usage: python collect.py <base_dir> <node_str>")
        exit(1)
    base_dir = sys.argv[1]
    node_str = sys.argv[2]
    print(aggregrate(base_dir, node_str))
