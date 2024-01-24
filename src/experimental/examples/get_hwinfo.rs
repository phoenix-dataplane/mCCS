use std::ffi::CString;

use cuda_runtime_sys::cudaDeviceGetPCIBusId;
use nvml_sys::{nvmlDeviceGetCpuAffinity, nvmlDeviceGetHandleByPciBusId_v2, nvmlInit_v2};

fn main() {
    let bus_id = CString::new(b"00000000:00:00.0").unwrap();
    let raw_bus_id = bus_id.as_c_str();
    // including the null terminator
    let len = raw_bus_id.to_bytes().len() + 1;
    let device = 0;
    unsafe {
        cudaDeviceGetPCIBusId(raw_bus_id.as_ptr() as *mut _, len as i32, device);
        let mut handle = std::ptr::null_mut();
        nvmlInit_v2();
        nvmlDeviceGetHandleByPciBusId_v2(raw_bus_id.as_ptr() as *mut _, &mut handle);
        let mut cpu_set = 0u64;
        nvmlDeviceGetCpuAffinity(handle, 1, &mut cpu_set);
        println!("CPU set for device {}: {:#066b}", device, cpu_set);
    }
}
