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


def convert_size(size: str):
    if size[-1] == "K":
        return int(size[:-1]) * 1024
    elif size[-1] == "M":
        return int(size[:-1]) * 1024 * 1024
    elif size[-1] == "G":
        return int(size[:-1]) * 1024 * 1024 * 1024
    else:
        return int(size)


class BenchArgs:
    def __init__(
        self,
        name: str,
        root_addr: str,
        rank: int,
        num_ranks: int,
        cuda_dev: int,
        size: int,
        comm: str,
        round: int,
    ) -> None:
        self.name = name
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
--communicator {self.comm} --round {self.round} --size-in-byte --name {self.name} --epoch 4"


def get_args_group(
    name: str,
    root_addr: str,
    rank_map: list,
    size: int,
    round: int,
    comm: int = 42,
):
    """
    rank_map: list of (machine_id, local_gpu_cnt)
    """
    args_group = []
    global_rank_cnt = 0
    num_ranks = sum([len(local_gpu_list) for _, local_gpu_list in rank_map])
    for machine, local_gpu_list in rank_map:
        for local_rank, gpu_idx in enumerate(local_gpu_list):
            args_group.append(
                (
                    machine,
                    BenchArgs(
                        name=name,
                        root_addr=root_addr,
                        rank=global_rank_cnt + local_rank,
                        num_ranks=num_ranks,
                        cuda_dev=gpu_idx,
                        size=size,
                        comm=comm,
                        round=round,
                    ),
                )
            )
        global_rank_cnt += len(local_gpu_list)
    return args_group


def gen_daemon(
    machine_id: int,
    daemon_args: str,
):
    return {
        "host": f"danyang-0{machine_id}",
        "bin": "mccs",
        "args": f"--host {machine_id} {daemon_args}",
        "weak": True,
        "dependencies": [],
    }


class AppProperties:
    def __init__(
        self, name: str, binary: str, size: str, rank_map: list, comm: int
    ) -> None:
        self.name = name
        self.binary = binary
        self.size = size
        self.rank_map = rank_map
        self.comm = comm


def generate_config(
    name: str,
    group: str,
    app_list,
    daemon_args: str,
    round: int = 25,
) -> dict:
    # get unique set of machines from app_list.machine_map
    machines = set()
    for app in app_list:
        machines.update([machine for machine, _ in app.rank_map])
    machines = list(machines)
    # generate daemons
    daemons = [gen_daemon(mid, daemon_args) for mid in machines]
    # generate apps
    apps = []
    for app in app_list:
        for machine, arg in get_args_group(
            app.name,
            addrs[app.rank_map[0][0]],
            app.rank_map,
            convert_size(app.size),
            comm=app.comm,
            round=round,
        ):
            apps.append(
                {
                    "host": f"danyang-0{machine}",
                    "bin": app.binary,
                    "args": arg.get_args(),
                    "dependencies": list(range(len(daemons))),
                }
            )
    # generate config
    config = {
        "name": name,
        "group": group,
        "worker": daemons + apps,
    }
    return config


def allreduce_setup1():
    job1_rank_map = [(2, [0, 1]), (1, [0, 1])]
    job2_rank_map = [(3, [0, 1]), (5, [0, 1])]
    config = generate_config(
        "multi-allreduce-ecmp-setup1",
        "multi-allreduce-ecmp",
        [
            AppProperties(
                name="app1",
                binary="allreduce_bench",
                size="128M",
                rank_map=job1_rank_map,
                comm=81,
            ),
            AppProperties(
                name="app2",
                binary="allreduce_bench",
                size="128M",
                rank_map=job2_rank_map,
                comm=82,
            ),
        ],
        "--config eval/multi-app/ecmp-setup1.toml",
    )
    with open("output/multi-allreduce-ecmp-setup1.toml", "w") as f:
        toml.dump(config, f)
    config = generate_config(
        "multi-allreduce-flow-setup1",
        "multi-allreduce-flow",
        [
            AppProperties(
                name="app1",
                binary="allreduce_bench",
                size="128M",
                rank_map=job1_rank_map,
                comm=81,
            ),
            AppProperties(
                name="app2",
                binary="allreduce_bench",
                size="128M",
                rank_map=job2_rank_map,
                comm=82,
            ),
        ],
        "--config eval/multi-app/flow-setup1.toml",
    )
    with open("output/multi-allreduce-flow-setup1.toml", "w") as f:
        toml.dump(config, f)


def allreduce_setup2():
    job1_rank_map = [(2, [0]), (3, [0]), (1, [0]), (5, [0])]
    job2_rank_map = [(2, [1]), (1, [1])]
    job3_rank_map = [(3, [1]), (5, [1])]
    config = generate_config(
        "multi-allreduce-ecmp-setup2",
        "multi-allreduce-ecmp",
        [
            AppProperties(
                name="app1",
                binary="allreduce_bench",
                size="128M",
                rank_map=job1_rank_map,
                comm=81,
            ),
            AppProperties(
                name="app2",
                binary="allreduce_bench",
                size="128M",
                rank_map=job2_rank_map,
                comm=82,
            ),
            AppProperties(
                name="app3",
                binary="allreduce_bench",
                size="128M",
                rank_map=job3_rank_map,
                comm=83,
            ),
        ],
        "--config eval/multi-app/ecmp-setup2.toml",
    )
    with open("output/multi-allreduce-ecmp-setup2.toml", "w") as f:
        toml.dump(config, f)
    config = generate_config(
        "multi-allreduce-flow-setup2",
        "multi-allreduce-flow",
        [
            AppProperties(
                name="app1",
                binary="allreduce_bench",
                size="128M",
                rank_map=job1_rank_map,
                comm=81,
            ),
            AppProperties(
                name="app2",
                binary="allreduce_bench",
                size="128M",
                rank_map=job2_rank_map,
                comm=82,
            ),
            AppProperties(
                name="app3",
                binary="allreduce_bench",
                size="128M",
                rank_map=job3_rank_map,
                comm=83,
            ),
        ],
        "--config eval/multi-app/flow-setup2.toml",
    )
    with open("output/multi-allreduce-flow-setup2.toml", "w") as f:
        toml.dump(config, f)


def allreduce_setup3():
    job1_rank_map = [(2, [0]), (3, [0]), (1, [0]), (5, [0])]
    job2_rank_map = [(2, [1]), (3, [1]), (1, [1]), (5, [1])]
    config = generate_config(
        "multi-allreduce-ecmp-setup3",
        "multi-allreduce-ecmp",
        [
            AppProperties(
                name="app1",
                binary="allreduce_bench",
                size="128M",
                rank_map=job1_rank_map,
                comm=81,
            ),
            AppProperties(
                name="app2",
                binary="allreduce_bench",
                size="128M",
                rank_map=job2_rank_map,
                comm=82,
            ),
        ],
        "--config eval/multi-app/ecmp-setup3.toml",
    )
    with open("output/multi-allreduce-ecmp-setup3.toml", "w") as f:
        toml.dump(config, f)
    config = generate_config(
        "multi-allreduce-flow-setup3",
        "multi-allreduce-flow",
        [
            AppProperties(
                name="app1",
                binary="allreduce_bench",
                size="128M",
                rank_map=job1_rank_map,
                comm=81,
            ),
            AppProperties(
                name="app2",
                binary="allreduce_bench",
                size="128M",
                rank_map=job2_rank_map,
                comm=82,
            ),
        ],
        "--config eval/multi-app/flow-setup3.toml",
    )
    with open("output/multi-allreduce-flow-setup3.toml", "w") as f:
        toml.dump(config, f)


def allreduce_setup4():
    job1_rank_map = [(2, [0, 1]), (1, [0, 1])]
    job2_rank_map = [(3, [0]), (5, [0])]
    job3_rank_map = [(3, [1]), (5, [1])]
    config = generate_config(
        "multi-allreduce-ecmp-setup4",
        "multi-allreduce-ecmp",
        [
            AppProperties(
                name="app1",
                binary="allreduce_bench",
                size="128M",
                rank_map=job1_rank_map,
                comm=81,
            ),
            AppProperties(
                name="app2",
                binary="allreduce_bench",
                size="128M",
                rank_map=job2_rank_map,
                comm=82,
            ),
            AppProperties(
                name="app3",
                binary="allreduce_bench",
                size="128M",
                rank_map=job3_rank_map,
                comm=83,
            ),
        ],
        "--config eval/multi-app/ecmp-setup4.toml",
    )
    with open("output/multi-allreduce-ecmp-setup4.toml", "w") as f:
        toml.dump(config, f)
    config = generate_config(
        "multi-allreduce-flow-setup4",
        "multi-allreduce-flow",
        [
            AppProperties(
                name="app1",
                binary="allreduce_bench",
                size="128M",
                rank_map=job1_rank_map,
                comm=81,
            ),
            AppProperties(
                name="app2",
                binary="allreduce_bench",
                size="128M",
                rank_map=job2_rank_map,
                comm=82,
            ),
            AppProperties(
                name="app3",
                binary="allreduce_bench",
                size="128M",
                rank_map=job3_rank_map,
                comm=83,
            ),
        ],
        "--config eval/multi-app/flow-setup4.toml",
    )
    with open("output/multi-allreduce-flow-setup4.toml", "w") as f:
        toml.dump(config, f)


def allreduce_reconfig():
    gpt_map = [(2, [0, 1]), (3, [0, 1]), (1, [0, 1]), (5, [0, 1])]

    config = generate_config(
        f"8gpu-dynamic-allreduce",
        f"8gpu-dynamic-allreduce",
        [
            AppProperties(
                name="reconfig-allreduce",
                binary="allreduce_bench",
                size="128M",
                rank_map=gpt_map,
                comm=600,
            )
        ],
        "--config eval/dynamic-config/reconfig.toml",
        round=1
    )
    with open(f"../dynamic-config/launch-allreduce-ring-reconfig.toml", "w") as f:
        toml.dump(config, f)


allreduce_setup1()
allreduce_setup2()
allreduce_setup3()
allreduce_setup4()
allreduce_reconfig()