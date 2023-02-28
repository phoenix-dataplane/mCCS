template<typename T>
struct FuncSum {
  __device__ FuncSum(uint64_t opArg=0) {}
  __device__ T operator()(const T x, const T y) const {
    return x + y;
  }
};

template<typename T>
struct FuncProd {
  __device__ FuncProd(uint64_t opArg=0) {}
  __device__ T operator()(const T x, const T y) const {
    return x * y;
  }
};

template<typename T>
struct FuncMax {
  __device__ FuncMax(uint64_t opArg=0) {}
  __device__ T operator()(const T x, const T y) const {
    return (x < y) ? y : x;
  }
};

template<typename T>
struct FuncMin {
  __device__ FuncMin(uint64_t opArg=0) {}
  __device__ T operator()(const T x, const T y) const {
    return (x < y) ? x : y;
  }
};