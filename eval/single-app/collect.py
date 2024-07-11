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
            return line.split(": ")[-1].split(" GB")[0]
    print("Error: no rank 0 found: ", lines)


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


def aggregrate_ecmp(n_gpu):
    base_dir_prefix = "/tmp/single-app"
    node_str = str(n_gpu) + "GPU_ECMP"
    results = []
    for i in range(10):
        base_dir = base_dir_prefix + str(i)
        if not os.path.exists(base_dir):
            continue
        results.append(aggregrate(base_dir, node_str))
    for i, result in enumerate(results):
        res = "Solution,App,Size (Bytes),Dtype,Latency (us),AlgBW (GB/s),BusBW (GB/s)\n"
        for command in result:
            for size in result[command]:
                res += f"{n_gpu}GPU_ECMP,{command},{convert_size(size)},float16,0,{result[command][size]},0\n"
        with open(f"./output/mccs-{n_gpu}gpu-ecmp/round{i}.csv", "w") as f:
            f.write(res)


def aggregate_flow(n_gpu):
    base_dir = "/tmp/single-app-flow"
    node_str = str(n_gpu) + "GPU_FLOW"
    result = aggregrate(base_dir, node_str)
    res = "Solution,App,Size (Bytes),Dtype,Latency (us),AlgBW (GB/s),BusBW (GB/s)\n"
    for command in result:
        for size in result[command]:
            res += f"{n_gpu}GPU_ECMP,{command},{convert_size(size)},float16,0,{result[command][size]},0\n"
    with open(f"./output/mccs-{n_gpu}gpu-flow/round0.csv", "w") as f:
        f.write(res)


def pick_any():
    if __name__ == "__main__":
        # read base dir and node_str from command line
        import sys

        if len(sys.argv) != 3:
            print("Usage: python collect.py <base_dir> <node_str>")
            exit(1)
        base_dir = sys.argv[1]
        node_str = sys.argv[2]
        print(aggregrate(base_dir, node_str))


def get_all_ecmp():
    if __name__ == "__main__":
        # read base dir and node_str from command line
        import sys

        if len(sys.argv) != 2:
            print("Usage: python collect.py <n_gpu>")
            exit(1)
        n_gpu = sys.argv[1]
        aggregrate_ecmp(n_gpu)


def get_all_flow():
    if __name__ == "__main__":
        # read base dir and node_str from command line
        import sys

        if len(sys.argv) != 2:
            print("Usage: python collect.py <n_gpu>")
            exit(1)
        n_gpu = sys.argv[1]
        aggregate_flow(n_gpu)


def collect_all():
    res = "Solution,App,Size (Bytes),Dtype,Latency (us),AlgBW (GB/s),BusBW (GB/s)"
    allgather_4gpu = [res]
    allgather_8gpu = [res]
    allreduce_4gpu = [res]
    allreduce_8gpu = [res]
    for i in range(10):
        base_dir = "/tmp/single-app" + str(i)
        if not os.path.exists(base_dir):
            continue
        node_str = "4GPU_ECMP"
        r = aggregrate(base_dir, node_str)
        for command in ["allgather", "allreduce"]:
            for size in r[command]:
                line = f"4GPU_ECMP,{command},{convert_size(size)},float16,0,{r[command][size]},0"
                if command == "allgather":
                    allgather_4gpu.append(line)
                else:
                    allreduce_4gpu.append(line)
        node_str = "8GPU_ECMP"
        r = aggregrate(base_dir, node_str)
        for command in ["allgather", "allreduce"]:
            for size in r[command]:
                line = f"8GPU_ECMP,{command},{convert_size(size)},float16,0,{r[command][size]},0"
                if command == "allgather":
                    allgather_8gpu.append(line)
                else:
                    allreduce_8gpu.append(line)
    base_dir = "/tmp/single-app-flow"
    node_str = "4GPU_FLOW"
    r = aggregrate(base_dir, node_str)
    for command in ["allgather", "allreduce"]:
        for size in r[command]:
            line = f"4GPU_FLOW,{command},{convert_size(size)},float16,0,{r[command][size]},0"
            if command == "allgather":
                allgather_4gpu.append(line)
            else:
                allreduce_4gpu.append(line)
    node_str = "8GPU_FLOW"
    r = aggregrate(base_dir, node_str)
    for command in ["allgather", "allreduce"]:
        for size in r[command]:
            line = f"8GPU_FLOW,{command},{convert_size(size)},float16,0,{r[command][size]},0"
            if command == "allgather":
                allgather_8gpu.append(line)
            else:
                allreduce_8gpu.append(line)
    with open("../plot/data/mccs-allgather-4gpu.csv", "w") as f:
        f.write("\n".join(allgather_4gpu) + "\n")
    with open("../plot/data/mccs-allgather-8gpu.csv", "w") as f:
        f.write("\n".join(allgather_8gpu) + "\n")
    with open("../plot/data/mccs-allreduce-4gpu.csv", "w") as f:
        f.write("\n".join(allreduce_4gpu) + "\n")
    with open("../plot/data/mccs-allreduce-8gpu.csv", "w") as f:
        f.write("\n".join(allreduce_8gpu) + "\n")


# get_all_ecmp()
# pick_any()
# get_all_flow()
collect_all()
