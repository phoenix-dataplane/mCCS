#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <getopt.h>
#include <sys/time.h>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <chrono>
#include <thread>

#include <cuda_runtime.h>

#include "nccl.h"
#include "mpi.h"

#include "toml.hpp"

#define MPICHECK(cmd) do {                         \
  int e = cmd;                            \
  if (e != MPI_SUCCESS) {                         \
    printf("Failed: MPI error %s:%d '%d'\n",       \
        __FILE__,__LINE__, e); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define HIPCHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: CUDA error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

/**
 * @brief Print configuration information
 * @return void
*/
static void print_config();
/**
 * @brief Get a hash of the hostname
 * @param hostname Hostname to hash
 * @return Hash of the hostname
*/
static uint64_t getHostHash(const char* hostname);
/**
 * @brief Get the hostname
 * @param hostname Buffer to store hostname
 * @param maxlen Maximum length of hostname
 * @return void
*/
static void getHostName(char *hostname, int maxlen);
/**
 * @brief Print usage information
 * @param progname Name of the program
 * @return void
*/
static void print_usage(const char *progname);
/**
 * @brief Parse command line arguments
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 * @return true if arguments are valid, false otherwise
*/
static bool parse_args(int argc, char *argv[]);

#define DEFAULT_ITERS 100
#define DEFAULT_WARMUP_ITERS 5

// Command line arguments
// Number of iterations
static unsigned int iters = DEFAULT_ITERS;
// Path to the folder of trace files
static char *trace_path = NULL;
// Verbose output
static bool verbose = false;
// Save path
static char *save_path = NULL;

// Communication operation types
enum CommOperationType {
    AllReduceOp,
    AllGatherOp,
};

// Communication operation specification
struct CommOperationSpec {
    // Communication operation type (e.g. AllReduce, Broadcast, etc.)
    int op_type;
    // Message size in bytes
    size_t msg_size;
    // compute interval
    uint64_t compute_interval;
};

// Workload specification
struct WorkloadSpec {
    std::vector<int> cuda_devices;
    // List of communication operations
    std::vector<CommOperationSpec> traces;
};

/**
 * @brief Load workload from a TOML file
 * @param trace_file Path to the trace file to load
 * @throw Exception if failed to load trace file
 * @return Workload specification
 */
WorkloadSpec load_workload(std::string trace_file)
{
    toml::table tbl;
    try
    {
        tbl = toml::parse_file(trace_file);
        std::cout << tbl << "\n";
    }
    catch (const toml::parse_error& err)
    {
        std::cerr << "Parsing failed:\n" << err << "\n";
    }

    std::vector<int> cuda_devices;
    std::vector<CommOperationSpec> trace_specs;

    auto devs = tbl["cuda_devices"];
    toml::array& arr = *devs.as_array();
    for (auto&& elem: arr) {
        if constexpr (toml::is_integer<decltype(elem)>) {
            cuda_devices.push_back(**(elem.as_integer()));
        } else {
            throw "Failed to parse trace file: invalid cuda_devices, not an array of integers.";
        }
    }
    

    auto traces = tbl["traces"];
    arr = *traces.as_array();
    for (auto&& elem: arr) { 
        for (auto&& op_trace: arr) {
            if constexpr (toml::is_table<decltype(op_trace)>) {
                auto& op_table = *op_trace.as_table();
                size_t msg_size = op_table["size"].value<int64_t>().value();
                std::string op_name = op_table["type"].value<std::string>().value();
                printf("test\n");
                
                int op_type = -1;
                if (op_name.compare("all_gather") == 0) {
                    op_type = AllGatherOp;
                } else if (op_name.compare("all_reduce") == 0) {
                    op_type = AllReduceOp;
                } else {
                    throw "Failed to parse trace file: unsupported op type.";
                }

                uint64_t compute_interval = op_table["compute_interval"].value<int64_t>().value();
                CommOperationSpec trace_spec = {
                    .op_type = op_type,
                    .msg_size = msg_size,
                    .compute_interval = compute_interval,
                }; 
                trace_specs.push_back(trace_spec);
            } else {
                throw "Failed to parse trace file: invalid trace specs, not an array of tables.";
            }
        }
    }

    WorkloadSpec workload = {
        .cuda_devices = std::move(cuda_devices),
        .traces = std::move(trace_specs),
    };

    return workload;
}

int main(int argc, char *argv[])
{
    int world_my_rank, world_num_ranks, world_local_rank;
    struct timeval ts_start, ts_end;

    // Parse command line arguments
    if (!parse_args(argc, argv)) {
        exit(EXIT_FAILURE);
    }

    // Initialize MPI
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_my_rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_num_ranks));

    if (verbose) {
        printf("[MPI Rank %d] Start!\n", world_my_rank);
        print_config();
    }

    std::string trace_file(trace_path);

    auto workload = load_workload(trace_file);

    if (verbose) {
        printf("[MPI Rank %d] Successfully load trace file %s.\n",
            world_my_rank, trace_file.c_str());
    }

    // Get local rank based on hostname
    uint64_t *hostHashs = new uint64_t[world_num_ranks];
    if (hostHashs == NULL) {
        fprintf(stderr, "ERROR: Failed to allocate hostHashs\n");
        exit(EXIT_FAILURE);
    }

    char hostname[256];
    getHostName(hostname, 256);
    hostHashs[world_my_rank] = getHostHash(hostname);
    // Exchange host hashes
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t),
        MPI_BYTE, MPI_COMM_WORLD));

    world_local_rank = 0;
    for (int i = 0; i < world_num_ranks; i++) {
        if (i == world_my_rank) {
            break;
        }
        // Count number of ranks with the same host hash before me
        // and use that as local rank
        if (hostHashs[i] == hostHashs[world_my_rank]) {
            world_local_rank++;
        }
    }
    int cuda_dev = workload.cuda_devices[world_my_rank];
    if (verbose) {
        printf("[MPI Rank %d] host %s, local rank %d\n, CUDA dev %d", world_my_rank, hostname, world_local_rank, cuda_dev);
    }


    // Check if number of GPUs is valid
    int num_available_gpus = 0;
    HIPCHECK(cudaGetDeviceCount(&num_available_gpus));

    if (world_local_rank + 1 > num_available_gpus) {
        fprintf(stderr, "ERROR: no enough GPUs available for local rank %d on host %s\n",
            world_local_rank, hostname);
        exit(EXIT_FAILURE);
    }

    ncclComm_t comm;
    ncclUniqueId id;

    cudaStream_t stream;
    HIPCHECK(cudaSetDevice(cuda_dev));
    HIPCHECK(cudaStreamCreate(&stream));

    // Initialize a NCCL communicator including all ranks for warmup
    if (world_my_rank == 0) {
        NCCLCHECK(ncclGetUniqueId(&id));
    }
    MPICHECK(MPI_Bcast((void*)(&id), sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    NCCLCHECK(ncclGroupStart());
    printf("[MPI Rank %d] NCCL warmup.\n", world_my_rank);
    ncclCommInitRank(&comm, world_num_ranks, id, world_my_rank);
    NCCLCHECK(ncclGroupEnd());
    printf("[MPI Rank %d] NCCL communicator group initialized, world rank %d.\n", world_my_rank, world_my_rank);
    HIPCHECK(cudaStreamSynchronize(stream));

    // Allocate memory for GPU buffers, streams and communicators
    float *devSendBuff;
    float *devRecvBuff;

    // Calculate the maximum buffer size needed for all the communication operations
    unsigned long int buffer_size = 0;
    for (auto& trace: workload.traces) {
        size_t req_size = trace.msg_size;
        if (trace.op_type == AllGatherOp) {
            req_size *= world_num_ranks;
        }
        if (req_size > buffer_size) {
            buffer_size = req_size;
        }
    }

    HIPCHECK(cudaMalloc((void**)&devSendBuff, buffer_size));
    HIPCHECK(cudaMalloc((void**)&devRecvBuff, buffer_size));
    HIPCHECK(cudaMemset(devSendBuff, 1, buffer_size));
    HIPCHECK(cudaMemset(devRecvBuff, 0, buffer_size));

    // Warm up
    printf("[MPI Rank %d] Warmup start\n", world_my_rank);

    unsigned int warmup_iters = DEFAULT_WARMUP_ITERS;
    for (unsigned int i = 0; i < warmup_iters; i++) {
        for (auto& trace : workload.traces) {
            NCCLCHECK(ncclGroupStart());
            switch (trace.op_type) {
                case AllGatherOp:
                    ncclAllGather((const void*)devSendBuff, (void*)devRecvBuff, trace.msg_size,
                        ncclInt8, comm, stream);
                    break;
                case AllReduceOp:
                    ncclAllReduce((const void*)devSendBuff, (void*)devRecvBuff, trace.msg_size,
                        ncclInt8, ncclSum, comm, stream);
                    break;

            }
            NCCLCHECK(ncclGroupEnd());
        }
        if (verbose) {
            printf("[MPI Rank %d][Warmup Iter %u] Success on host %s\n", world_my_rank, i, hostname);
        }
    }

    // Wait for completion
    HIPCHECK(cudaStreamSynchronize(stream));
    printf("[MPI Rank %d] Warmup complete\n", world_my_rank);

    // Record start time
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> iter_times;
    auto start = std::chrono::high_resolution_clock::now();
    // Communicate using NCCL
    for (unsigned int i = 0; i < iters; i++) {
        for (auto& trace : workload.traces) {
            // sleep 
            std::this_thread::sleep_for(std::chrono::microseconds(trace.compute_interval));
            NCCLCHECK(ncclGroupStart());
            switch (trace.op_type) {
                case AllGatherOp:
                    ncclAllGather((const void*)devSendBuff, (void*)devRecvBuff, trace.msg_size,
                        ncclInt8, comm, stream);
                    break;
                case AllReduceOp:
                    ncclAllReduce((const void*)devSendBuff, (void*)devRecvBuff, trace.msg_size,
                        ncclInt8, ncclSum, comm, stream);
                    break;

            }
            NCCLCHECK(ncclGroupEnd());
        }
        if (verbose) {
            printf("[MPI Rank %d][Iter %u] Success on host %s\n", world_my_rank, i, hostname);
        }
        HIPCHECK(cudaStreamSynchronize(stream));
        auto end = std::chrono::high_resolution_clock::now();
        auto iter_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        iter_times.push_back(iter_time);
        if (verbose && world_my_rank == 0) {
            printf("Iter time: %ld ms\n", iter_time);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

    if (world_my_rank == 0) {
        if (save_path != NULL) {
            std::ofstream save_file(save_path);
            if (save_file.is_open()) {
                for (auto& iter_time: iter_times) {
                    save_file << iter_time << "\n";
                }
                save_file.close();
            } else {
                fprintf(stderr, "ERROR: Failed to open save file %s\n", save_path);
            }
        }
    }

    // Free device buffers and streams
    HIPCHECK(cudaFree(devSendBuff));
    HIPCHECK(cudaFree(devRecvBuff));
    HIPCHECK(cudaStreamDestroy(stream));

    NCCLCHECK(ncclCommDestroy(comm));

    // Finalize MPI
    MPICHECK(MPI_Finalize());

    return 0;
}

/**
 * Print configuration
 * @return void
 */
static void print_config()
{
    printf("Number of iterations: %u\n", iters);
    printf("Trace path: %s\n", trace_path);
    printf("Verbose: %s\n", verbose ? "true" : "false");
}

/**
 * Get hash of hostname
 * @param hostname Hostname
 * @return Hash of hostname
 */
static uint64_t getHostHash(const char* hostname)
{
    // Based on DJB2 hash function
    uint64_t hash = 5381;

    for (int i = 0; hostname[i] != '\0'; i++) {
        hash = ((hash << 5) + hash) + hostname[i];
    }

    return hash;
}

/**
 * Get hostname
 * @param hostname Hostname
 * @param maxlen Maximum length of hostname
 * Note: This function truncates the hostname at the first '.' character
 */
static void getHostName(char *hostname, int maxlen)
{
    gethostname(hostname, maxlen);

    for (int i = 0; i < maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}

/**
 * Parse command line arguments
 * @param argc Number of arguments
 * @param argv Array of arguments
 * @return true if arguments are valid, false otherwise
 */
static bool parse_args(int argc, char *argv[])
{
    while (1) {
        static struct option long_options[] = {
            {"iters",           required_argument,  0,  'n'},
            {"trace_path",      required_argument,  0,  'p'},
            {"save_path",       no_argument,        0,  's'},
            {"verbose",         no_argument,        0,  'v'},
            {"help",            no_argument,        0,  'h'},
            {0,                 0,                  0,  0}
        };

        int option_index = 0;
        int c = getopt_long(argc, argv, "n:p:g:vsh", long_options, &option_index);

        if (c == -1) {
            break;
        }

        switch (c) {
            case 'n':
                iters = atoi(optarg);
                if (iters == 0) {
                    fprintf(stderr, "ERROR: Invalid number of iterations %s\n", optarg);
                    print_usage(argv[0]);
                }
                break;

            case 'p':
                trace_path = optarg;
                break;

            case 's':
                save_path = optarg;
                break;

            case 'v':
                verbose = true;
                break;

            case 'h':
                print_usage(argv[0]);
                exit(EXIT_SUCCESS);

            default:
                fprintf(stderr, "ERROR: Invalid option %c\n", c);
                print_usage(argv[0]);
                return false;
        }
    }

    if (optind < argc) {
        fprintf(stderr, "ERROR: Invalid argument %s\n", argv[optind]);
        print_usage(argv[0]);
        return false;
    }

    return true;
}

/**
 * Print usage information
 * @param progname Name of the program
 * @return None
 */
static void print_usage(const char *progname)
{
    fprintf(stderr, "Usage: %s [OPTIONS]\n", progname);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -n, --iters <iters>      Number of iterations (default: %u)\n", DEFAULT_ITERS);
    fprintf(stderr, "  -p, --trace_path <path>  Path to the workload file\n");
    fprintf(stderr, "  -s, --save_path <path>   Path to the save iteration time\n");
    fprintf(stderr, "  -v, --verbose            Print verbose output\n");
    fprintf(stderr, "  -h, --help               Print this help message\n");
}
