use std::cell::RefCell;
use std::sync::atomic::Ordering;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU8, AtomicUsize};
use std::thread;

use spin::Mutex;

use cuda_runtime_sys::cudaSetDevice;

use super::CoreMask;
use super::EngineContainer;
use crate::cuda_warning;
use crate::engine::EngineStatus;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct RuntimeId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RuntimeMode {
    Dedicated = 0,
    Shared = 1,
}

pub struct Runtime {
    pub running: RefCell<Vec<EngineContainer>>,
    pub cuda_dev: AtomicI32,
    pub cores: CoreMask,

    pub active_cnt: AtomicUsize,
    pub mode: AtomicU8,

    pub new_pending: AtomicBool,
    pub pending: Mutex<Vec<EngineContainer>>,
}

unsafe impl Sync for Runtime {}

impl Runtime {
    pub fn new(cuda_dev: Option<i32>, cores: CoreMask, mode: RuntimeMode) -> Self {
        let cuda_dev = match cuda_dev {
            Some(dev) => {
                assert!(dev >= 0, "Invalid cuda device index");
                dev
            }
            None => -1,
        };
        Runtime {
            running: RefCell::new(Vec::new()),
            cuda_dev: AtomicI32::new(cuda_dev),
            cores,
            active_cnt: AtomicUsize::new(0),
            mode: AtomicU8::new(mode as u8),
            new_pending: AtomicBool::new(false),
            pending: Mutex::new(Vec::new()),
        }
    }
}

impl Runtime {
    pub fn mainloop(&self) {
        let cuda_idx = self.cuda_dev.load(Ordering::Relaxed);
        if cuda_idx != -1 {
            unsafe {
                cuda_warning!(cudaSetDevice(cuda_idx as i32));
            }
        }

        let mut shutdown = Vec::new();
        loop {
            let mut engines = self.running.borrow_mut();
            for (idx, engine) in engines.iter_mut().enumerate() {
                match engine.progress() {
                    EngineStatus::Idle => {}
                    EngineStatus::Progressed => {}
                    EngineStatus::Completed => {
                        shutdown.push(idx);
                    }
                }
            }

            self.active_cnt.fetch_sub(shutdown.len(), Ordering::Relaxed);
            for engine_idx in shutdown.drain(..).rev() {
                let engine = engines.swap_remove(engine_idx);
                std::mem::drop(engine);
            }

            if Ok(true)
                == self.new_pending.compare_exchange(
                    true,
                    false,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                )
            {
                let cuda_dev = self.cuda_dev.load(Ordering::Relaxed);
                if cuda_idx != -1 {
                    unsafe {
                        cuda_warning!(cudaSetDevice(cuda_idx as i32));
                    }
                }
                let mut pending_engines = self.pending.lock();
                let num_pendings = pending_engines.len();
                self.active_cnt.fetch_add(num_pendings, Ordering::Relaxed);
                engines.append(&mut pending_engines);
            } else {
                if engines.is_empty() {
                    log::trace!("Runtime shutting down...");
                    thread::park();
                    log::trace!("Runtime restaring...");
                }
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        let active = self.active_cnt.load(Ordering::Relaxed) == 0;
        let pending = self.pending.lock().is_empty();
        active && pending
    }

    pub fn add_engine(&self, engine: EngineContainer) {
        self.pending.lock().push(engine);
        self.new_pending.store(true, Ordering::Release);
    }

    #[inline]
    pub fn try_acquire(
        &self,
        mode: RuntimeMode,
        cuda_dev: Option<i32>,
        cores: CoreMask,
        quota: Option<usize>,
    ) -> bool {
        if self.cores != cores {
            return false;
        }
        let scheduled_engines = self.active_cnt.load(Ordering::Relaxed) + self.pending.lock().len();
        if self.is_empty() {
            self.mode.store(mode as u8, Ordering::Relaxed);
            if let Some(dev) = cuda_dev {
                assert!(dev >= 0, "Invalid cuda device index");
                self.cuda_dev.store(dev, Ordering::Relaxed);
            } else {
                self.cuda_dev.store(-1, Ordering::Relaxed);
            }
            true
        } else {
            let curr_mode = self.mode.load(Ordering::Relaxed);
            if curr_mode != mode as u8 {
                false
            } else {
                let curr_dev = self.cuda_dev.load(Ordering::Relaxed);
                if let Some(dev) = cuda_dev {
                    if curr_dev == -1 {
                        self.cuda_dev.store(dev, Ordering::Relaxed);
                    } else if curr_dev != dev {
                        return false;
                    }
                }
                match mode {
                    RuntimeMode::Dedicated => false,
                    RuntimeMode::Shared => {
                        if let Some(quota) = quota {
                            quota > scheduled_engines
                        } else {
                            true
                        }
                    }
                }
            }
        }
    }
}
