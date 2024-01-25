# %%
import toml

addrs = [
    "0.0.0.0",
    "192.168.211.2",
    "192.168.211.34",
    "192.168.211.66",
    "192.168.211.130",
    "192.168.211.162",
    "192.168.211.195",
]


class BenchArgs:
    def __init__(
        self,
        root_addr: str,
        rank: int,
        num_ranks: int,
        cuda_dev: int,
        size: int,
        comm: str,
        round: int,
    ) -> None:
        self.root_addr = root_addr
        self.rank = rank
        self.num_ranks = num_ranks
        self.cuda_dev = cuda_dev
        self.size = size
        self.comm = comm
        self.round = round

    def get_args(self):
        return f"--root-addr {self.root_addr} --rank {self.rank} \
--num-ranks {self.num_ranks} --cuda-device-idx {self.cuda_dev} --size {self.size} \
--communicator {self.comm} --round {self.round} --size_in_byte true"


def get_args_group(
    root_addr: str, machine_map: map, size: int, round: int = 10, comm: int = 42
):
    """
    machine_map: a list of machine number to how many ranks this machine uses
    """
    args_group = []
    global_rank_cnt = 0
    num_ranks = sum(machine_map.values())
    for machine, local_gpu_cnt in machine_map.items():
        for local_rank in range(local_gpu_cnt):
            args_group.append(
                (
                    machine,
                    BenchArgs(
                        root_addr=root_addr,
                        rank=global_rank_cnt + local_rank,
                        num_ranks=num_ranks,
                        cuda_dev=local_rank,
                        size=size,
                        comm=comm,
                        round=round,
                    ),
                )
            )
        global_rank_cnt += local_gpu_cnt
    return args_group


def generate_config(
    name: str, group: str, binary: str, root_addr: str, machine_map: map, size: int
) -> dict:
    def gen_daemon(machine_id: int):
        return {
            "host": f"danyang-0{machine_id}",
            "bin": "mccs",
            "args": f"--host {machine_id}",
            "dependencies": [],
        }

    workers = [gen_daemon(machine) for machine in machine_map.keys()]
    dep = [i for i in range(len(machine_map))]
    for machine, arg in get_args_group(root_addr, machine_map, size):
        workers.append(
            {
                "host": f"danyang-0{machine}",
                "bin": binary,
                "args": arg.get_args(),
                "dependencies": dep,
            }
        )

    config = {
        "name": name,
        "group": group,
        "workers": workers,
    }
    return config


def convert_size(size: str):
    if size[-1] == "K":
        return int(size[:-1]) * 1024
    elif size[-1] == "M":
        return int(size[:-1]) * 1024 * 1024
    elif size[-1] == "G":
        return int(size[:-1]) * 1024 * 1024 * 1024
    else:
        return int(size)


size_list = ["1K", "4K", "16K", "64K", "256K", "1M", "4M", "16M", "64M", "256M", "1G"]
command = ["allreduce", "allgather"]
node_configurations = [{1: 2, 2: 2, 3: 2, 5: 2}, {1: 1, 2: 1, 3: 1, 5: 1}]

for comm in command:
    for node_idx, node_config in enumerate(node_configurations):
        node_str = "8GPU" if node_idx == 0 else "4GPU"
        for size in size_list:
            config = generate_config(
                name=f"{comm}/{node_str}/{size}",
                group="microbench",
                binary=comm + "_bench",
                root_addr=addrs[1],
                machine_map=node_config,
                size=convert_size(size),
            )
            with open(f"output/{comm}_{node_str}_{size}.toml", "w") as f:
                toml.dump(config, f)
