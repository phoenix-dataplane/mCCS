use std::fmt;
use std::num::NonZeroUsize;

pub struct DeviceHostPtr<T> {
    host_ptr: *const T,
    dev_ptr: *const T,
}

impl<T> DeviceHostPtr<T> {
    #[inline]
    pub unsafe fn new_unchecked(ptr_host: *mut T, ptr_dev: *mut T) -> Self {
        DeviceHostPtr {
            host_ptr: ptr_host,
            dev_ptr: ptr_dev,
        }
    }

    #[inline]
    pub fn new(ptr_host: *mut T, ptr_dev: *mut T) -> Option<Self> {
        if !ptr_host.is_null() && !ptr_dev.is_null() {
            Some(unsafe { Self::new_unchecked(ptr_host, ptr_dev) })
        } else {
            None
        }
    }

    #[must_use]
    #[inline]
    pub fn addr_host(&self) -> NonZeroUsize {
        unsafe { NonZeroUsize::new_unchecked(self.host_ptr.addr()) }
    }

    #[must_use]
    #[inline]
    pub fn addr_dev(&self) -> NonZeroUsize {
        unsafe { NonZeroUsize::new_unchecked(self.dev_ptr.addr()) }
    }

    #[must_use]
    #[inline]
    pub fn as_ptr_host(&self) -> *mut T {
        self.host_ptr as *mut T
    }

    #[must_use]
    #[inline]
    pub fn as_ptr_dev(&self) -> *mut T {
        self.dev_ptr as *mut T
    }

    #[inline]
    pub fn cast<U>(self) -> DeviceHostPtr<U> {
        unsafe {
            DeviceHostPtr::new_unchecked(self.as_ptr_host() as *mut U, self.as_ptr_dev() as *mut U)
        }
    }
}

impl<T> Clone for DeviceHostPtr<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for DeviceHostPtr<T> {}

impl<T> fmt::Debug for DeviceHostPtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DeviceHostMapped")
            .field(&self.as_ptr_host())
            .field(&self.as_ptr_dev())
            .finish()
    }
}

impl<T> fmt::Pointer for DeviceHostPtr<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DeviceHostMapped")
            .field(&self.as_ptr_host())
            .field(&self.as_ptr_dev())
            .finish()
    }
}
