use std::fmt;
use std::num::NonZeroUsize;

#[repr(transparent)]
pub struct DeviceNonNull<T> {
    pointer: *const T,
}

impl<T> DeviceNonNull<T> {
    pub const unsafe fn new_unchecked(ptr: *mut T) -> Self {
        DeviceNonNull { pointer: ptr as _ }
    }

    #[inline]
    pub fn new(ptr: *mut T) -> Option<Self> {
        if !ptr.is_null() {
            Some(unsafe { Self::new_unchecked(ptr) })
        } else {
            None
        }
    }

    #[inline]
    pub fn addr(self) -> NonZeroUsize {
        unsafe { NonZeroUsize::new_unchecked(self.pointer.addr()) }
    }

    #[inline]
    pub fn with_addr(self, addr: NonZeroUsize) -> Self {
        unsafe { Self::new_unchecked(self.pointer.with_addr(addr.get()) as *mut _) }
    }

    #[inline]
    pub fn map_addr(self, f: impl FnOnce(NonZeroUsize) -> NonZeroUsize) -> Self {
        self.with_addr(f(self.addr()))
    }

    #[inline]
    pub const fn as_ptr(self) -> *mut T {
        self.pointer as *mut T
    }

    #[inline]
    pub const fn cast<U>(self) -> DeviceNonNull<U> {
        unsafe { DeviceNonNull::new_unchecked(self.as_ptr() as *mut U) }
    }
}

impl<T> Clone for DeviceNonNull<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for DeviceNonNull<T> {}

impl<T> fmt::Debug for DeviceNonNull<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DeviceNonNull")
            .field(&self.as_ptr())
            .finish()
    }
}

impl<T> fmt::Pointer for DeviceNonNull<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DeviceNonNull")
            .field(&self.as_ptr())
            .finish()
    }
}
