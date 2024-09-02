# Overview of MCCS's Code Structure
Here we provide an overview of MCCS's overall code structure. All MCCS's implementation can be found in `src` folder. `launcher` folder contains a simple distributed launcher (similar to PDSH's functionality) to launch MCCS backends and benchmark applications for evaluation, with configuration files contained in `eval` folder.

The major components under `src` folder are:
- `collectives`: CUDA kernels for `AllGather` and `AllReduce` using host memory transport or RDMA transport; adapted from NCCL.
- `ipc`: Rust library for shared memory queue communication between MCCS shim library and MCCS service.
- `libmccs:` MCCS's shim library used by applications.
- `mccs`: MCCS service
- `collectives-sys, cuda-sys, gdrcopy-sys, ibverbs`: Rust bindings for launching the collectives kernels, CUDA, GDRCopy and ibverbs.

## MCCS Service
The main control logic of MCCS service is implemented in `src/mccs/src/control.rs`. It spins up proxy engines and an exchange engine. The control thread (main thread) listens for new applications and creates a daemon engine for each new user (application) thread.

MCCS engines are similar to asynchronous future executors that asynchronously process some input commands. Each MCCS engine can be scheduled on a runtime (thread). The runtime is implemented in `src/mccs/src/runtime`, where we currently put each MCCS engine on a dedicated runtime (thread). However, multiple engines can be scheduled on the same runtime to save CPU; each runtime will progress all the engines scheduled to it in a round robin fashion.

### Daemon Engines
Daemon engines are implemented in `src/mccs/src/daemon`. Daemon engines are used to connect with the application's MCCS shim library. Each application thread corresponds to a daemon engine. The daemon engines establish shared memory queues to the applications, which are used by the shim library to send memory allocation or collectives requests and receive replies (IPC memory handles and IPC events which are used for stream synchronization). The daemon engine relays the collective requests to the proxy engines, which schedules collective kernels to GPUs. The daemon engine also handles CUDA memory allocations and returns the IPC memory handles to the applications.

### Proxy Engines (section 4.2 in paper)
Proxy engines are implemented in `src/mccs/src/proxy`. Each GPU have a single corresponding proxy engine. The proxy engines are responsible for bridging the gap between high-level communicators and low-level resources. Proxy engines setups transport resources, e.g., host memory buffers shared by GPU and transport engines. Proxy engines schedule the launch of collective CUDA kernels, capture the contents of the internal stream for collectives in IPC events. Proxy engines also schedule tasks for transport engines. Control of collective strategies and network resources can be imposed in proxy engines.

### Transport Engines (section 4.2 in paper)
Implemented in `src/mccs/src/transport`. Transport engines are used for inter-host communication. For intra host communication using e.g., host memory, no CPU intervation is necessary, as GPU can directly read/write the shared host memory buffers. However, for inter-host transport using RDMA, we need to spin up some CPU threads to issue RDMA writes when GPU finishes copying to host memory buffers, or write to host memory buffers for GPU to process when a chunk has been transferred from remote. Transport engines are used to do such work. There is no strict correlation between a transport engine and a communicator. Each transport engine can handle work from multiple communicators and user applications. The number of transport engines (CPU threads) can also be dynamically adjusted. Since transport engines issue all RDMA operations, fine-grained control of network traffic can be imposed in transport engines.

### Bootstrap
Implemented in `src/mccs/src/bootstrap`. To setup or reconfigure a communicator, new RDMA connections need to be setup depending on the collective algorithm (e.g., ring ordering). Following NCCL's implementation, we let each rank communicates with the root (i.e., rank 0) to form an AllGather TCP/IP-based ring and use this ring to exchange bootstrap information to setup resources like RDMA connections.

### Exchange Engine
There will be a single exchange engine for each MCCS service instance. Exchange engines are used by MCCS services to exchange control path information between MCCS services on different hosts. Bootstrap handles (e.g., root rank's TCP listening socket port) are sent to all other ranks by the exchange engine. Network administer's policies (e.g., reconfigure an application) will also be received by the exchange engine and delivered to corresponding proxy engines.

## MCCS Shim Library
`src/libmccs/src` contains the implementation for the shim library. To initialize the communicator, we provide the following API:
```rust
pub fn init_communicator_rank(
    unique_id: u32,
    rank: usize,
    num_ranks: usize,
    cuda_device_idx: i32,
    root_addr: IpAddr,
) -> Result<MccsCommunicatorHandle, Error> 
```
which is similar to the one provided by NCCL
```c
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank)
```
Note that in NCCL, current device set for CUDA is used, while our API allows the users to explicitly specify a device.   
Before a CUDA stream is first used, `register_stream(cuda_dev: i32, stream: cudaStream_t)` should be called so an IPC event will be created for this stream to share with MCCS service. MCCS service can then synchronizes with the application stream using the IPC event.   

Then, the users can use collective APIs such as  the following:
```rust
pub fn all_gather(
    comm: MccsCommunicatorHandle,
    send_buf: DevicePtr,
    recv_buf: DevicePtr,
    size: usize,
    stream: cudaStream_t,
) -> Result<(), Error> 
```
which is also similar to NCCL's:
```c
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream)
```
