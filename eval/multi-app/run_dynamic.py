import os
def file_paths(base_dir: str):
    res = []
    for group in os.readdir(base_dir):
        for each in os.readdir(os.path.join(base_dir, group)):
            if each.startswith("traffic_gen"):
                res.append(os.path.join(base_dir, group, each))
    return res

def parse_file(path: str):
    with open(path, "r") as f:
        lines = f.readlines()
    res = []
    for line in lines:
        if line.find("[Rank 0] Iter") != -1:
            prefix = line.split("[")[0]
            iter_no = line.split("Iter ")[-1].split(" ")[0]
            time_ms = line.split("time: ")[-1].split(" ")[0]
            res.append((prefix, iter_no, time_ms))
    return res