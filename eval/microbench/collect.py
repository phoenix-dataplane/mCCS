import os

size_list = ["1K", "4K", "16K", "64K", "256K", "1M", "4M", "16M", "64M", "256M", "1G"]
command = ["allreduce", "allgather"]
node_str = ["8GPU", "4GPU"]

def get_output(base_dir:str, command:str, node_str:str, size:str):
    path = os.path.join(base_dir, command, node_str, size)
    read_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".stdout") and f.find("bench") != -1]
    file_contents = []
    for f in read_files:
        with open(f, "r") as f:
            file_contents.append(f.read())
    return file_contents

def filter_contents(contents:list):
    def filter(lines:str):
        return [line for line in lines.split("\n") if line.find("bandwidth") != -1]
    res = []
    for i in [filter(content) for content in contents]:
        res.extend(i)
    return res
    

def rank0_parse_bw(lines:list):
    for line in lines:
        if line.find("Rank 0") != -1:
            return line.split(": ")[-1]

print(rank0_parse_bw(filter_contents(get_output("/tmp/results", "allgather", "8GPU", "1M"))))