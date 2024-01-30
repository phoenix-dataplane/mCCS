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
    vgg_map = [(2, 1), (3, 1), (1, 1), (5, 1)]
    gpt1_map = [(2, 1), (1, 1)]
    gpt2_map = [(3, 1), (5, 1)]

    config = generate_traffic_gen_config(
        "setup2-vgg",
        "qos",
        [
            TraceProperties(
                name="vgg",
                config="workloads/setup-2_vgg.toml",
                rank_map=vgg_map,
                iter_cnt=51,
            ),
            # TraceProperties(
            #     name="gpt_1",
            #     config="workloads/setup-2_gpt_1.toml",
            #     rank_map=gpt1_map,
            #     iter_cnt=501,
            # ),
            # TraceProperties(
            #     name="gpt_2",
            #     config="workloads/setup-2_gpt_2.toml",
            #     rank_map=gpt2_map,
            #     iter_cnt=501,
            # ),
        ],
        # "--config eval/multi-app/output/setup2-mccs-config.toml",
        # "--config eval/multi-app/setup2-trace-fair.toml",
        "--config eval/multi-app/setup2-trace-qosv1.toml",
    )

    with open("output/setup2-vgg-qos.toml", "w") as f:
        toml.dump(config, f)


def setup1_profile():
    job1_map = [(2, 2), (1, 2)]

    config = generate_traffic_gen_config(
        "setup1-profile",
        "setup1-profile",
        [
            TraceProperties(
                name="job1",
                config="workloads/setup-1_vgg_0.toml",
                rank_map=job1_map,
                iter_cnt=101,
            ),
        ],
        "--config eval/multi-app/setup1-trace-profile.toml",
    )

    with open("output/setup1-trace-profile.toml", "w") as f:
        toml.dump(config, f)


def setup1_fair():
    job1_map = [(2, 2), (1, 2)]
    job2_map = [(3, 2), (5, 2)]

    config = generate_traffic_gen_config(
        "setup1-fair",
        "setup1-fair",
        [
            TraceProperties(
                name="job1",
                config="workloads/setup-1_vgg_0.toml",
                rank_map=job1_map,
                iter_cnt=101,
            ),
            TraceProperties(
                name="job2",
                config="workloads/setup-1_vgg_1.toml",
                rank_map=job2_map,
                iter_cnt=101,
            ),
        ],
        "--config eval/multi-app/setup1-trace-fair.toml",
    )

    with open("output/setup1-trace-fair.toml", "w") as f:
        toml.dump(config, f)


def setup4_real_qos(setup: str, is_ecmp: bool = False):
    vgg_map = [(2, 2), (1, 2)]
    gpt1_map = [(3, 1), (5, 1)]
    gpt2_map = [(3, 1), (5, 1)]
    if is_ecmp:
        ecmp_str = "-ecmp"
    else:
        ecmp_str = ""

    if setup == "qosv2":
        config = f"eval/multi-app/output/setup4-mccs{ecmp_str}-config.toml"
    else:
        config = f"eval/multi-app/setup4-trace{ecmp_str}-{setup}.toml"

    config = generate_traffic_gen_config(
        f"setup4-real{ecmp_str}-{setup}",
        f"setup4-real{ecmp_str}",
        [
            TraceProperties(
                name="vgg",
                config="workloads/setup-4_vgg.toml",
                rank_map=vgg_map,
                iter_cnt=101,
            ),
            TraceProperties(
                name="gpt_1",
                config="workloads/setup-4_gpt_1.toml",
                rank_map=gpt1_map,
                iter_cnt=1501,
            ),
            TraceProperties(
                name="gpt_2",
                config="workloads/setup-4_gpt_2.toml",
                rank_map=gpt2_map,
                iter_cnt=3001,
            ),
        ],
        "--config " + config,
    )

    with open(f"output/setup4-real{ecmp_str}-{setup}.toml", "w") as f:
        toml.dump(config, f)


def setup4_dynamic():
    vgg_map = [(2, 2), (1, 2)]
    gpt1_map = [(3, 1), (5, 1)]
    gpt2_map = [(3, 1), (5, 1)]

    config = generate_traffic_gen_config(
        f"setup4-dynamic",
        f"setup4-dynamic",
        [
            TraceProperties(
                name="vgg",
                config="workloads/setup-4_vgg.toml",
                rank_map=vgg_map,
                iter_cnt=5001,
            ),
            TraceProperties(
                name="gpt_1",
                config="workloads/setup-4_gpt_1.toml",
                rank_map=gpt1_map,
                iter_cnt=40001,
            ),
            TraceProperties(
                name="gpt_2",
                config="workloads/setup-4_gpt_2.toml",
                rank_map=gpt2_map,
                iter_cnt=40001,
            ),
        ],
        "--config eval/dynamic-config/setup4-trace-fair.toml",
    )

    config["worker"], worker_1, worker_2 = (
        config["worker"][:8],
        config["worker"][8:10],
        config["worker"][10:12],
    )
    for i in worker_1:
        i["dependencies"] = []
    for i in worker_2:
        i["dependencies"] = []
    launch_gpt_1 = {
        "name": "setup4-dynamic-gpt-1",
        "group": "setup4-dynamic-gpt-1",
        "worker": worker_1,
    }
    launch_gpt_2 = {
        "name": "setup4-dynamic-gpt-2",
        "group": "setup4-dynamic-gpt-2",
        "worker": worker_2,
    }

    with open(f"../dynamic-config/launch.toml", "w") as f:
        toml.dump(config, f)
    with open(f"../dynamic-config/launch-gpt-1.toml", "w") as f:
        toml.dump(launch_gpt_1, f)
    with open(f"../dynamic-config/launch-gpt-2.toml", "w") as f:
        toml.dump(launch_gpt_2, f)


def ring_reconfig():
    gpt_map = [(2, 2), (3, 2), (1, 2), (5, 2)]

    config = generate_traffic_gen_config(
        f"setup4-dynamic",
        f"setup4-dynamic",
        [
            TraceProperties(
                name="gpt",
                config="workloads/reconfig_gpt.toml",
                rank_map=gpt_map,
                iter_cnt=5001,
            ),
        ],
        "--config eval/dynamic-config/reconfig.toml",
    )
    with open(f"../dynamic-config/launch-ring-reconfig.toml", "w") as f:
        toml.dump(config, f)


setup2_vgg_qos()
setup1_profile()
setup1_fair()
setup4_real_qos("fair")
setup4_real_qos("qosv1")
setup4_real_qos("qosv2")
setup4_real_qos("fair", True)
setup4_real_qos("qosv1", True)
setup4_real_qos("qosv2", True)
setup4_dynamic()
ring_reconfig()
