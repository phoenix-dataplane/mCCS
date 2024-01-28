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


class TraceProperties:
    def __init__(self, name: str, config: str, rank_map: list, iter_cnt: int) -> None:
        self.name = name
        self.binary = "traffic_gen"
        self.config = config
        self.rank_map = rank_map
        self.iter_cnt = iter_cnt


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


class RealTrafficArgs:
    def __init__(
        self,
        name: str,
        root_addr: str,
        rank: int,
        iter: int,
        config: str,
    ) -> None:
        self.name = name
        self.root_addr = root_addr
        self.rank = rank
        self.iter = iter
        self.config = config

    def get_args(self) -> str:
        return f"--root-addr {self.root_addr} --rank {self.rank} \
--iters {self.iter} --config {self.config} --verbose --name {self.name}"


def get_traffic_gen_groups(
    name: str,
    rank_map: list,
    root_addr: str,
    iter: int,
    config: str,
) -> list[(int, RealTrafficArgs)]:
    res = []
    rank = 0
    for machine, local_rank_cnt in rank_map:
        for _ in range(local_rank_cnt):
            res.append(
                (
                    machine,
                    RealTrafficArgs(
                        name=name,
                        root_addr=root_addr,
                        rank=rank,
                        iter=iter,
                        config=config,
                    ),
                )
            )
            rank += 1
    return res


def generate_traffic_gen_config(
    name: str,
    group: str,
    app_list: list[TraceProperties],
    daemon_args: str,
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
        for machine, arg in get_traffic_gen_groups(
            app.name,
            app.rank_map,
            addrs[app.rank_map[0][0]],
            app.iter_cnt,
            app.config,
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


def setup2_vgg_qos():
    vgg_map = [(2, 1), (1, 1)]
    resnet_map = [(3, 1), (5, 1)]
    gpt_map = [(2, 1), (3, 1), (1, 1), (5, 1)]

    config = generate_traffic_gen_config(
        "setup2-vgg",
        "qos",
        [
            TraceProperties(
                name="vgg",
                config="workloads/setup-2_vgg.toml",
                rank_map=vgg_map,
                iter_cnt=50,
            ),
            TraceProperties(
                name="resnet",
                config="workloads/setup-2_resnet.toml",
                rank_map=resnet_map,
                iter_cnt=50,
            ),
            TraceProperties(
                name="gpt",
                config="workloads/setup-2_gpt.toml",
                rank_map=gpt_map,
                iter_cnt=50,
            ),
        ],
        "--config eval/multi-app/setup2-vgg.toml",
    )

    with open("output/setup2-vgg-qos.toml", "w") as f:
        toml.dump(config, f)


setup2_vgg_qos()
