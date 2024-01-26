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
--communicator {self.comm} --round {self.round} --size-in-byte"


def get_args_group(
    root_addr: str, machine_map: list, size: int, round: int = 10, comm: int = 42
):
    """
    machine_map: list of (machine_id, local_gpu_cnt)
    """
    args_group = []
    global_rank_cnt = 0
    num_ranks = sum([local_gpu_cnt for _, local_gpu_cnt in machine_map])
    for machine, local_gpu_cnt in machine_map:
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
    name: str,
    group: str,
    binary: str,
    root_addr: str,
    machine_map: list,
    size: int,
    comm: int = 42,
    daemon_args: str = "",
) -> dict:
    def gen_daemon(machine_id: int):
        return {
            "host": f"danyang-0{machine_id}",
            "bin": "mccs",
            "args": f"--host {machine_id} {daemon_args}",
            "weak": True,
            "dependencies": [],
        }

    workers = [gen_daemon(machine[0]) for machine in machine_map]
    dep = [i for i in range(len(machine_map))]
    for machine, arg in get_args_group(
        root_addr, machine_map, size, round=20, comm=comm
    ):
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
        "worker": workers,
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


def generate(size_list, command, node_configurations, communicator, mccs_cfg_path):
    for comm in command:
        for node_configs in node_configurations:
            node_str, node_config = node_configs
            root = node_config[0][0]
            for size in size_list:
                config = generate_config(
                    name=f"{node_str}/{comm}/{size}",
                    group=node_str,
                    binary=comm + "_bench",
                    root_addr=addrs[root],
                    machine_map=node_config,
                    size=convert_size(size),
                    comm=communicator,
                    daemon_args=f"--config {mccs_cfg_path}",
                )
                with open(f"output/{node_str}_{comm}_{size}.toml", "w") as f:
                    toml.dump(config, f)


def four_gpu_ecmp():
    size_list = ["32K", "128K", "512K", "2M", "8M", "32M", "128M", "512M"]
    command = ["allreduce", "allgather"]
    node_configurations = [
        ("4GPU_ECMP", [(2, 1), (3, 1), (1, 1), (5, 1)]),
    ]
    generate(size_list, command, node_configurations, 42, "eval/single-app/4gpu.toml")

def four_gpu_flow_scheduling():
    size_list = ["32K", "128K", "512K", "2M", "8M", "32M", "128M", "512M"]
    command = ["allreduce", "allgather"]
    node_configurations = [
        ("4GPU_FLOW", [(2, 1), (3, 1), (1, 1), (5, 1)]),
    ]
    generate(size_list, command, node_configurations, 137, "eval/single-app/4gpu.toml")

def eight_gpu_ecmp():
    size_list = ["32K", "128K", "512K", "2M", "8M", "32M", "128M", "512M"]
    command = ["allreduce", "allgather"]
    node_configurations = [
        ("8GPU_ECMP", [(2, 2), (3, 2), (1, 2), (5, 2)]),
    ]
    generate(size_list, command, node_configurations, 42, "eval/single-app/8gpu.toml")

def eight_gpu_flow_scheduling():
    size_list = ["32K", "128K", "512K", "2M", "8M", "32M", "128M", "512M"]
    command = ["allreduce", "allgather"]
    node_configurations = [
        ("8GPU_FLOW", [(2, 2), (3, 2), (1, 2), (5, 2)]),
    ]
    generate(size_list, command, node_configurations, 137, "eval/single-app/8gpu.toml")

four_gpu_ecmp()
four_gpu_flow_scheduling()
eight_gpu_ecmp()
eight_gpu_flow_scheduling()
