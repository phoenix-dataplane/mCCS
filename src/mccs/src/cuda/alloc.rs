use std::ffi::c_void;
use std::num::NonZeroUsize;

use crate::cuda_warning;
use cuda_runtime_sys::{
    cudaFree, cudaFreeHost, cudaHostAlloc, cudaHostGetDevicePointer, cudaHostRegister,
    cudaHostUnregister, cudaMalloc,
};
use cuda_runtime_sys::{cudaGetDevice, cudaSetDevice};
use cuda_runtime_sys::{cudaHostAllocMapped, cudaHostRegisterMapped};

use super::mapped_ptr::DeviceHostPtr;
use super::ptr::DeviceNonNull;

#[derive(Clone, Copy, Debug)]
enum MappedType {
    Alloc,
    Register,
}

pub struct DeviceHostMapped<T> {
    ptr: DeviceHostPtr<T>,
    size: usize,
    device: i32,
    ty: MappedType,
}

impl<T> DeviceHostMapped<T> {
    pub fn alloc(count: usize) -> Self {
        let size = count * std::mem::size_of::<T>();
        let mut ptr_host: *mut c_void = std::ptr::null_mut();
        let mut ptr_dev: *mut c_void = std::ptr::null_mut();
        let mut device = 0;
        unsafe {
            cudaGetDevice(&mut device);
            cuda_warning!(cudaHostAlloc(&mut ptr_host, size, cudaHostAllocMapped));
            cuda_warning!(cudaHostGetDevicePointer(&mut ptr_dev, ptr_host, 0));
        }
        let ptr = unsafe { DeviceHostPtr::new_unchecked(ptr_host as *mut T, ptr_dev as *mut T) };
        DeviceHostMapped {
            ptr,
            size,
            device,
            ty: MappedType::Alloc,
        }
    }

    pub fn register(ptr_host: *mut T, count: usize) -> Option<Self> {
        if !ptr_host.is_null() {
            let size = count * std::mem::size_of::<T>();
            let mut ptr_dev: *mut c_void = std::ptr::null_mut();
            let mut device = 0;
            unsafe {
                cudaGetDevice(&mut device);
                cudaHostRegister(ptr_host as *mut _, size, cudaHostRegisterMapped);
                cudaHostGetDevicePointer(&mut ptr_dev, ptr_host as *mut _, 0);
            }
            log::info!("DeviceHostMapped register: {:p}", ptr_dev);
            let ptr = unsafe { DeviceHostPtr::new_unchecked(ptr_host, ptr_dev as *mut T) };
            let mapped = DeviceHostMapped {
                ptr,
                size,
                device,
                ty: MappedType::Register,
            };
            Some(mapped)
        } else {
            None
        }
    }

    #[must_use]
    #[inline]
    pub fn addr_host(&self) -> NonZeroUsize {
        self.ptr.addr_host()
    }

    #[must_use]
    #[inline]
    pub fn addr_dev(&self) -> NonZeroUsize {
        self.ptr.addr_dev()
    }

    #[must_use]
    #[inline]
    pub fn as_ptr_host(&self) -> *mut T {
        self.ptr.as_ptr_host()
    }

    #[must_use]
    #[inline]
    pub fn as_ptr_dev(&self) -> *mut T {
        self.ptr.as_ptr_dev()
    }

    #[inline]
    pub fn cast<U>(self) -> DeviceHostMapped<U> {
        let wrapper = std::mem::ManuallyDrop::new(self);
        let ptr = wrapper.ptr.cast();
        DeviceHostMapped {
            ptr,
            size: wrapper.size,
            device: wrapper.device,
            ty: wrapper.ty,
        }
    }

    /// Returns the number of bytes in the mapped allocation/registration
    #[must_use]
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }
}

impl<T> Drop for DeviceHostMapped<T> {
    fn drop(&mut self) {
        let mut curr_dev = 0;
        unsafe {
            cudaGetDevice(&mut curr_dev);
            if curr_dev != self.device {
                cudaSetDevice(self.device);
            }
        }
        match self.ty {
            MappedType::Alloc => unsafe {
                cudaFreeHost(self.as_ptr_host() as *mut _);
            },
            MappedType::Register => unsafe {
                cudaHostUnregister(self.as_ptr_host() as *mut _);
            },
        }
    }
}

pub struct DeviceAlloc<T> {
    ptr: DeviceNonNull<T>,
    size: usize,
    device: i32,
}

impl<T> DeviceAlloc<T> {
    pub fn new(count: usize) -> Self {
        let size = count * std::mem::size_of::<T>();
        let mut device = 0;
        let mut dev_ptr: *mut c_void = std::ptr::null_mut();
        unsafe {
            cudaGetDevice(&mut device);
            cudaMalloc(&mut dev_ptr, size);
        }
        let ptr = unsafe { DeviceNonNull::new_unchecked(dev_ptr as *mut T) };
        DeviceAlloc { ptr, size, device }
    }

    #[must_use]
    #[inline]
    pub fn addr(&self) -> NonZeroUsize {
        self.ptr.addr()
    }

    #[must_use]
    #[inline]
    pub fn as_ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }

    #[must_use]
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }
}

impl<T> Drop for DeviceAlloc<T> {
    fn drop(&mut self) {
        let mut curr_dev = 0;
        unsafe {
            cudaGetDevice(&mut curr_dev);
            if curr_dev != self.device {
                cudaSetDevice(self.device);
            }
            cudaFree(self.as_ptr() as *mut _);
        }
    }
}
