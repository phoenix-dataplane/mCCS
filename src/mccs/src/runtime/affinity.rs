use std::ffi::CString;
use std::fmt;

use cuda_runtime_sys::cudaDeviceGetPCIBusId;
use nvml_sys::{nvmlDeviceGetCpuAffinity, nvmlDeviceGetHandleByPciBusId_v2, nvmlInit_v2};

pub(crate) use libnuma::masks::CpuMask;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoreMask(CpuMask);

unsafe impl Send for CoreMask {}

macro_rules! nvml_warning {
    ($nvml_op:expr) => {{
        let e = $nvml_op;
        if e != nvml_sys::nvmlReturn_enum::NVML_SUCCESS {
            log::error!("NVML failed with {:?} at {}:{}.", e, file!(), line!())
        }
    }};
}

pub fn init_nvml() {
    unsafe {
        nvml_warning!(nvml_sys::nvmlInit_v2());
    }
}

impl CoreMask {
    pub(crate) fn from_numa_node(numa_node_affinity: Option<u8>) -> Self {
        use libnuma::masks::indices::{CpuIndex, NodeIndex};
        use libnuma::masks::Mask;
        let all_cpus = CpuIndex::number_of_permitted_cpus();
        match numa_node_affinity {
            None => {
                // Do not use CpuMask::all()!!!!!
                let cpu_mask = CpuMask::allocate();
                for i in 0..all_cpus {
                    cpu_mask.set(CpuIndex::new(i as _));
                }
                CoreMask(cpu_mask)
            }
            Some(node) => {
                let mut node = NodeIndex::new(node);
                let cpu_mask = node.node_to_cpus();
                CoreMask(cpu_mask)
            }
        }
    }

    pub(crate) fn sched_set_affinity_for_current_thread(&self) -> bool {
        self.0.sched_set_affinity_for_current_thread()
    }

    pub(crate) fn from_cpu_set(cpu_set: u64) -> Self {
        use libnuma::masks::indices::CpuIndex;
        use libnuma::masks::Mask;
        let cpu_mask = CpuMask::allocate();
        for i in 0..64 {
            if cpu_set & (1 << i) != 0 {
                cpu_mask.set(CpuIndex::new(i as _));
            }
        }
        CoreMask(cpu_mask)
    }

    pub(crate) fn from_device_affinity(device_idx: i32) -> CoreMask {
        let bus_id = CString::new(b"00000000:00:00.0").unwrap();
        let raw_bus_id = bus_id.as_c_str();
        // including the null terminator
        let len = raw_bus_id.to_bytes().len() + 1;
        let cpu_set = unsafe {
            cudaDeviceGetPCIBusId(raw_bus_id.as_ptr() as *mut _, len as i32, device_idx);
            let mut handle = std::ptr::null_mut();
            nvml_warning!(nvmlDeviceGetHandleByPciBusId_v2(
                raw_bus_id.as_ptr() as *mut _,
                &mut handle
            ));
            let mut cpu_set = 0u64;
            nvml_warning!(nvmlDeviceGetCpuAffinity(handle, 1, &mut cpu_set));
            cpu_set
        };
        Self::from_cpu_set(cpu_set)
    }

    #[allow(unused)]
    pub(crate) fn is_set(&self, i: u16) -> bool {
        use libnuma::masks::indices::CpuIndex;
        use libnuma::masks::Mask;
        self.0.is_set(CpuIndex::new(i))
    }
}

impl fmt::Display for CoreMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use libnuma::masks::indices::CpuIndex;
        use libnuma::masks::Mask;
        let all_cpus = CpuIndex::number_of_permitted_cpus();
        let mut masks = Vec::with_capacity((all_cpus + 32 - 1) / 32);
        for i in (0..all_cpus).step_by(32) {
            let mut mask = 0;
            let nbits = (all_cpus - i).min(32);
            for j in 0..nbits {
                if self.0.is_set(CpuIndex::new((i + j) as u16)) {
                    mask |= 1 << j;
                }
            }
            if nbits == 32 {
                masks.push(format!("{:08x}", mask));
            } else {
                let mut mask_str = String::new();
                for j in 0..nbits / 4 {
                    mask_str.insert_str(0, &format!("{:0x}", mask >> (j * 4) & 0xf));
                }
                masks.push(mask_str);
            }
        }
        masks.reverse();
        write!(f, "{:?}", masks)
    }
}
