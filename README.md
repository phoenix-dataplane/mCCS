<h1 align="center">
MCCS
</h1>
<p align="center">Managed Collective Communication Service</p>
<p align="center">
<a href="./LICENSE">
<img src="https://img.shields.io/badge/license-Apache_2.0-blue.svg" alt="License: Apache 2.0">
</a>
</p>

## Prerequisites
- rust/cargo
- python3
- justfile


## Build Guide
1. Build CUDA code
Under `src/collectives`, run `make` to build the CUDA binaries.
2. Build Rust code
Under `src`, run `cargo build --release` to build the Rust code.

## Evaluation Guide
### Clean Up
In case of unexpected errors, run the following commands to terminate previous processes:
```bash
just killall
```
### Prepare Configuration Files
1. Under `eval/single-app` create directory `output`, and run the `gen_config.py`.
2. Under `eval/multi-app` create directory `output`, and run the `gen_config.py` and `gen_traffic_gen_config.py`.

### Single Application
Run the following commands respectively under the root directory of the project:
```bash
just four_gpu_ecmp
just eight_gpu_ecmp
just four_gpu_flow
just eight_gpu_flow
```

### Multi Application
Run the following commands respectively under the root directory of the project:
```bash
just batched-allreduce-multi-final
```