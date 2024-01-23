use std::alloc::{GlobalAlloc, Layout, System};
use std::ops::{Deref, DerefMut};

// Transport buffer is designed to be a host registerd buffer
// for GPU kernels to read/write
// agent may also access the memory
pub struct TransportBuffer<T> {
    ptr: *mut T,
    size: usize,
    align: usize,
}

unsafe impl<T> Send for TransportBuffer<T> {}
unsafe impl<T> Sync for TransportBuffer<T> {}

impl<T> TransportBuffer<T> {
    pub fn new(meta: T, size: usize, align: usize) -> TransportBuffer<T> {
        let layout = Layout::from_size_align(size, align).unwrap();
        let ptr = unsafe { System.alloc(layout) } as *mut T;
        assert_eq!(ptr.align_offset(std::mem::align_of::<T>()), 0);
        let aligned_size = layout.size();
        unsafe {
            nix::sys::mman::mlock(ptr as *mut _, aligned_size).unwrap();
            *ptr = meta;
        };
        TransportBuffer { ptr, size, align }
    }

    #[inline]
    pub unsafe fn get_meta(&self) -> &T {
        &*self.ptr
    }

    #[inline]
    pub unsafe fn get_meta_mut(&self) -> &mut T {
        &mut *self.ptr
    }

    #[inline]
    pub fn meta_ptr(&self) -> *const T {
        self.ptr as *const _
    }

    #[inline]
    pub fn meta_mut_ptr(&self) -> *mut T {
        self.ptr
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    #[inline]
    pub fn buf_ptr(&self) -> *const u8 {
        unsafe { self.ptr.add(1) as *const u8 }
    }

    #[inline]
    pub fn buf_mut_ptr(&self) -> *mut u8 {
        unsafe { self.ptr.add(1) as *mut u8 }
    }

    #[inline]
    pub fn buf_slice(&self) -> *const [u8] {
        unsafe {
            let buf_ptr = self.ptr.add(1) as *const _;
            let buf_size = self.size - std::mem::size_of::<T>();
            std::ptr::slice_from_raw_parts(buf_ptr, buf_size)
        }
    }

    #[inline]
    pub fn buf_mut_slice(&self) -> *mut [u8] {
        unsafe {
            let buf_ptr = self.ptr.add(1) as *mut _;
            let buf_size = self.size - std::mem::size_of::<T>();
            std::ptr::slice_from_raw_parts_mut(buf_ptr, buf_size)
        }
    }

    #[inline]
    pub fn buf_size(&self) -> usize {
        self.size - std::mem::size_of::<T>()
    }
}

impl<T> Deref for TransportBuffer<T> {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe {
            let buf_ptr = self.ptr.add(1) as *const _;
            let buf_size = self.size - std::mem::size_of::<T>();
            std::slice::from_raw_parts(buf_ptr, buf_size)
        }
    }
}

impl<T> DerefMut for TransportBuffer<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            let buf_ptr = self.ptr.add(1) as *mut _;
            let buf_size = self.size - std::mem::size_of::<T>();
            std::slice::from_raw_parts_mut(buf_ptr, buf_size)
        }
    }
}

impl<T> AsRef<[u8]> for TransportBuffer<T> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &*self
    }
}

impl<T> Drop for TransportBuffer<T> {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.size, self.align).unwrap();
        unsafe {
            std::ptr::drop_in_place(self.ptr);
            System.dealloc(self.ptr as *mut _, layout);
        }
    }
}
