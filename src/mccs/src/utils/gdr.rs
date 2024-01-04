use std::ffi::c_void;
use std::marker::PhantomData;

use log::trace;
use once_cell::sync::Lazy;

use cuda_driver_sys::CUdevice;
use cuda_driver_sys::{cuDeviceGet, cuDeviceGetAttribute, CUdevice_attribute};
use cuda_runtime_sys::cudaDriverGetVersion;

use gdrcopy_sys::gdr_pin_buffer;
use gdrcopy_sys::gdr_t;
use gdrcopy_sys::GPU_PAGE_MASK;
use gdrcopy_sys::GPU_PAGE_OFFSET;
use gdrcopy_sys::GPU_PAGE_SIZE;
use gdrcopy_sys::{gdr_close, gdr_open};
use gdrcopy_sys::{gdr_driver_get_version, gdr_runtime_get_version};
use gdrcopy_sys::{gdr_get_info_v2, gdr_map};
use gdrcopy_sys::{gdr_info_t, gdr_mh_t};
use gdrcopy_sys::{gdr_unmap, gdr_unpin_buffer};

use crate::cuda::alloc::DeviceAlloc;
use crate::utils::CU_INIT;
use crate::{cu_warning, cuda_warning};

macro_rules! gdr_warning {
    ($gdr_op:expr) => {{
        let e = $gdr_op;
        if e != 0 {
            log::error!(
                "GDRCOPY failed with error code {:?} at {}:{}.",
                e,
                file!(),
                line!()
            )
        }
    }};
}

macro_rules! align_size {
    ($size:ident,$align:expr) => {
        $size = (($size + ($align) - 1) / ($align)) * ($align);
    };
}

// GdrHandle is thread safe
// https://github.com/NVIDIA/gdrcopy/issues/152
pub(crate) struct GdrHandle(pub(crate) gdr_t);

unsafe impl Send for GdrHandle {}
unsafe impl Sync for GdrHandle {}

pub(crate) static GDR_HANDLE: Lazy<GdrHandle> = Lazy::new(|| gdr_init());

fn gdr_init() -> GdrHandle {
    let handle = unsafe { gdr_open() };

    if !handle.is_null() {
        let mut lib_major = 0;
        let mut lib_minor = 0;
        let mut drv_major = 0;
        let mut drv_minor = 0;
        unsafe {
            gdr_runtime_get_version(&mut lib_major, &mut lib_minor);
        }
        unsafe {
            gdr_driver_get_version(handle, &mut drv_major, &mut drv_minor);
        }
        if (lib_major < 2 || (lib_major == 2 && lib_minor < 1))
            || (drv_major < 2 || (drv_major == 2 && drv_minor < 1))
        {
            if !handle.is_null() {
                unsafe { gdr_warning!(gdr_close(handle)) };
            }
        }
    }
    GdrHandle(handle)
}

pub struct GdrMappedMem<T> {
    gdr_dev_mem: DeviceAlloc<u8>,
    gdr_map: *mut c_void,
    gdr_host_offset: isize,
    gdr_dev_offset: usize,
    gdr_map_size: usize,
    gdr_unaligned_size: usize,
    gdr_mh: gdr_mh_t,
    phantom: PhantomData<T>,
}

unsafe impl<T> Send for GdrMappedMem<T> {}
unsafe impl<T> Sync for GdrMappedMem<T> {}

impl<T> GdrMappedMem<T> {
    pub(crate) fn new(num_elems: usize) -> GdrMappedMem<T> {
        let mut map_size = std::mem::size_of::<T>() * num_elems;
        let unaligned_size = map_size;
        align_size!(map_size, GPU_PAGE_SIZE as usize);
        let dev_mem = DeviceAlloc::<u8>::new(map_size + GPU_PAGE_SIZE as usize - 1);
        let dev_ptr = dev_mem.as_ptr();
        let aligned_addr =
            unsafe { dev_ptr.add(GPU_PAGE_OFFSET as usize).addr() & GPU_PAGE_MASK as usize } as u64;
        let align = aligned_addr as usize - dev_ptr.addr();
        unsafe {
            let mut mh = std::mem::MaybeUninit::<gdr_mh_t>::uninit();
            gdr_warning!(gdr_pin_buffer(
                GDR_HANDLE.0,
                aligned_addr,
                map_size,
                0,
                0,
                mh.as_mut_ptr()
            ));
            let mh: gdrcopy_sys::gdr_mh_s = mh.assume_init();
            let mut map_va = std::ptr::null_mut::<c_void>();
            gdr_warning!(gdr_map(GDR_HANDLE.0, mh, &mut map_va, map_size));
            let mut info = std::mem::MaybeUninit::<gdr_info_t>::uninit();
            gdr_warning!(gdr_get_info_v2(GDR_HANDLE.0, mh, info.as_mut_ptr()));
            let info = info.assume_init();
            let offset = (info.va - aligned_addr) as isize;
            trace!(
                "
                GDRCOPY: allocated devMap {:p} gdrMap {:p} offset {:x} mh {:x} mapSize {} at {}",
                dev_mem.as_ptr(),
                map_va,
                offset + align as isize,
                mh.h,
                map_size,
                map_va.addr() + offset as usize
            );
            GdrMappedMem {
                gdr_dev_mem: dev_mem,
                gdr_map: map_va,
                gdr_host_offset: offset,
                gdr_dev_offset: (offset + align as isize) as usize,
                gdr_map_size: map_size,
                gdr_unaligned_size: unaligned_size,
                gdr_mh: mh,
                phantom: PhantomData,
            }
        }
    }

    #[inline]
    pub(crate) fn get_cpu_ptr(&self) -> *mut T {
        unsafe { ((self.gdr_map as *mut u8).offset(self.gdr_host_offset)) as *mut T }
    }

    #[inline]
    pub(crate) fn get_gpu_ptr(&self) -> *mut T {
        unsafe { (self.gdr_dev_mem.as_ptr().add(self.gdr_dev_offset)) as *mut T }
    }

    #[inline]
    pub(crate) fn get_unaligned_size(&self) -> usize {
        self.gdr_unaligned_size
    }
}

impl<T> Drop for GdrMappedMem<T> {
    fn drop(&mut self) {
        unsafe {
            gdr_warning!(gdr_unmap(
                GDR_HANDLE.0,
                self.gdr_mh,
                self.gdr_map,
                self.gdr_map_size
            ));
            gdr_warning!(gdr_unpin_buffer(GDR_HANDLE.0, self.gdr_mh));
        }
    }
}

pub fn check_dma_buf_support(cuda_device_idx: i32) -> bool {
    unsafe {
        CU_INIT.with(|_| {});
        let mut cuda_driver_version = 0;
        cuda_warning!(cudaDriverGetVersion(&mut cuda_driver_version));
        let mut dev = CUdevice::default();
        cu_warning!(cuDeviceGet(&mut dev, cuda_device_idx));
        let mut flag = 0;
        cu_warning!(cuDeviceGetAttribute(
            &mut flag,
            CUdevice_attribute::CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED,
            dev
        ));
        if flag > 0 {
            true
        } else {
            false
        }
    }
}

#[cfg(all(target_feature = "sse", target_arch = "x86_64"))]
pub fn wc_store_fence() {
    unsafe {
        core::arch::x86_64::_mm_sfence();
    }
}
