<h1 align="center">
MCCS Artifact
</h1>
<p align="center">Managed Collective Communication Service</p>
<p align="center">
<a href="./LICENSE">
<img src="https://img.shields.io/badge/license-Apache_2.0-blue.svg" alt="License: Apache 2.0">
</a>
</p>

This repo contains the source code for the SIGCOMM paper MCCS and scripts to reproduce major results in the paper (Figure 6, 8, 9).

## Environment
MCCS's evaluation requires configuration on the switch to create our evaluation scenarios. We have setup such environment on our on-premise testbed in Duke. Please leave your public key as a comment on HotCRP to let us give your access to the servers. We will setup a user for you and then provide you with instructions to access our testbed. 

Please coordinate with other reviewers to ensure that only one reviewer is conducting experiments on our servers at a time. Otherwise, the scripts may not work as expected.

The detailed setup of our environment can be found in [Environment Setup](docs/setup.md)

## Prerequisites
- rust/cargo
- python3
- [justfile](https://github.com/casey/just) 

## [Code Overview](docs/overview.md)

## Build Guide
1. Clone the code   
```
git clone https://github.com/phoenix-dataplane/mCCS.git
```
2. Build CUDA code   
Under `src/collectives`, run `make` to build the CUDA binaries.
```
cd src/collectives
make -j 
```
3. Build Rust code   
Under the project root directory, run `cargo build --release` to build the Rust code.
```
cargo build --release
```

## Evaluation Guide
### Clean Up
In case of unexpected errors (e.g., scripts are mistakenly launched), run the following commands to terminate previous processes:
```bash
just killall
```
Also, make sure no other processes with 'mccs' in the commandline run on the corresponding machines. 
Otherwise auto-clean functionality in the script may not work properly.:

### Prepare Configuration Files for Evaluations
Run the following command:
```bash
just prepare-environment
```

### Single Application Benchmarks
This step will reproduce Figure 6 for single application benchmarks.

First, run all MCCS benchmarks under ECMP and the best-fit flow scheduling algorithm from Section 4.3 of the paper.   
Run the following commands respectively under the root directory of the project:
```bash
just four_gpu_ecmp
just eight_gpu_ecmp 
just four_gpu_flow
just eight_gpu_flow
```
Collect the results for MCCS using:
```bash
cd eval/single-app
python collect.py
```
The results will be generated to `eval/plot/data` as CSV files `mccs-[allgather/allreduce]-[4/8]gpu.csv` for the 4 settings correspond to Figure 6 (a)-(d). 

We have also prepared the pre-run results for NCCL baselines under the CSV files in `eval/plot/data`. 
To plot the figures, just append the MCCS results in the 4 CSV files to the CSV files under `eval/plot/single_app`. Install `matplotlib`, `seaborn`, `pandas`, `numpy` and `scipy`.
```bash
pip install matplotlib seaborn pandas numpy scipy
cd eval/plot
python single_app/main.py
```
The figures for Figure 6 (a)-(d) will be generated under `eval/plot/figures` as `[allgather/allreduce]_[4/8]gpu.pdf`.

#### Running NCCL baselines (optional)
However, if you want to run NCCL baselines yourself, first build NCCL v2.17.1 using the official [NCCL repo](https://github.com/NVIDIA/nccl/tree/v2.17.1-1):
```bash
git clone https://github.com/NVIDIA/nccl.git
git checkout v2.17.1-1
make -j src.build CUDA_HOME=/usr/local/cuda-12.3 NVCC_GENCODE="-gencode=arch=compute_86,code=sm_86"
```

Then build our forked version of [nccl-tests repo](https://github.com/NVIDIA/nccl-tests.git) in `nccl-tests-mccs` folder, where the changes we made are provided in `nccl-test.patch` file.
```bash
cd nccl-tests-mccs
make -j MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi CUDA_HOME=/usr/local/cuda-12.3 NCCL_HOME=[path to NCCL v2.17.1 library just built]
```

Make sure the switch is set to ECMP setting by running:
```bash
./eval/set_ecmp_hashing_algo.sh everything
``` 

Then run NCCL and NCCL(OR) under all settings:
```bash
cd nccl-tests-mccs/microbenchmark
./one_click_run_nccl_all.sh
```

Collect the results using and paste the results to corresponding CSV files under `eval/plot/single_app` for the plot script.
```bash
python collect_nccl.py --app [allgather/allreduce] --num-gpus [1/2]
```

### Multi Application Benchmarks
This step will reproduce Figure 8 for the multiple application benchmarks.

First, run the following commands respectively under the root directory of the project to get the results of MCCS under all 4 settings. 
```bash
just batched-allreduce-multi-final
```
Collect the results:
```bash
cd eval/multi-app
python collect_multi.py
```
Append the results `multi-setting-[1/2/3/4].csv` under `eval/plot/data` to `setting[1/2/3/4].csv` under `eval/plot/multi_app`, where the pre-run NCCL baselines results are provided.   
To generate the figures, run:
```bash
cd eval/plot
python multi_app/main.py
```
The figures will be generated as `setting[1/2/3/4]_allreduce.pdf` under `eval/plot/figures`.

Note that in our scripts, setting 3 corresponds to Setup 4 in the paper, i.e., Figure 8(d); setting 4 corresponds to Setup 3 in the paper, i.e., Figure 8(c). 

#### Running NCCL baselines (optional)
The config files and scripts to run multi application benchmarks for NCCL and NCCL(OR) is in `nccl-test-mccs/setting[1-4]`. We take setting 2 for example.
```
cd eval/multi-app
# run benchmarks
./run_nccl_all_jobs_multiple_times.sh 20 [goodring/badring]   # NCCL(OR) / NCCL
# collect the results
python collect_nccl.py --solution [NCCL GR/NCCL BR]  --strip-head 2 --strip-tail 2  # NCCL(OR) / NCCL
```
Then paste the results to corresponding CSV files in `setting[1/2/3/4].csv` under `eval/plot/multi_app`.   
We note that NCCL(OR) and NCCL are equivalent for setting 1 and setting 4.


### Real Workloads
This step will reproduce Figure 9 in the paper.

Run the following commands respectively under the root directory of the project:
```bash
just collect-setup4-ecmp
just collect-setup4-normal
```
Collect the results:
```bash
cd eval/multi-app
python collect_real_workload.py
```
Copy the results generated in `eval/plot/data/real_workload.csv` to `eval/plot/real_workload/jct.csv` for the plotting script, where we have provided pre-run results. Plot the figure:
```
cd eval/plot
python real_workload/plot_jct.py
```
The figure will be generated to `eval/plot/figures/real_workload_jct.pdf`.
