use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread::{self, JoinHandle};

use super::CoreMask;
use super::{EngineContainer, Runtime, RuntimeId, RuntimeMode};
use crate::engine::SchedulingMode;

struct RuntimeHandle {
    id: RuntimeId,
    runtime: Arc<Runtime>,
    join_handle: JoinHandle<()>,
}

pub struct RuntimeManager {
    runtime_counter: AtomicU64,
    runtimes: Mutex<Vec<RuntimeHandle>>,
}

impl RuntimeManager {
    pub fn new() -> Self {
        RuntimeManager {
            runtime_counter: AtomicU64::new(0),
            runtimes: Mutex::new(Vec::new()),
        }
    }

    pub fn submit_engine(
        &self,
        engine: EngineContainer,
        cuda_dev: Option<i32>,
        cores: Option<CoreMask>,
    ) {
        let mode = engine.scheduling_hint();
        let runtime_mode = match mode {
            SchedulingMode::Dedicated => RuntimeMode::Dedicated,
            SchedulingMode::Compact => RuntimeMode::Shared,
        };
        let cores = match cores {
            None => CoreMask::from_numa_node(None),
            Some(c) => c,
        };
        let mut runtimes = self.runtimes.lock().unwrap();
        let r = runtimes.iter().find(|r| {
            r.runtime
                .try_acquire(runtime_mode, cuda_dev, cores.clone(), None)
        });
        if let Some(handle) = r {
            handle.runtime.add_engine(engine);
            handle.join_handle.thread().unpark();
        } else {
            let id = self.runtime_counter.fetch_add(1, Ordering::Relaxed);
            let runtime_id = RuntimeId(id);
            let handle = Self::start_runtime(runtime_id, cuda_dev, cores, runtime_mode);
            handle.runtime.add_engine(engine);
            handle.join_handle.thread().unpark();
            runtimes.push(handle);
        }
    }

    fn start_runtime(
        runtime_id: RuntimeId,
        cuda_dev: Option<i32>,
        cores: CoreMask,
        mode: RuntimeMode,
    ) -> RuntimeHandle {
        let runtime = Arc::new(Runtime::new(cuda_dev, cores.clone(), mode));
        let flag = runtime.try_acquire(mode, cuda_dev, cores.clone(), None);
        assert!(flag);

        let cuda_dev = cuda_dev.unwrap_or(-1);
        let runtime_thread = Arc::clone(&runtime);
        let join_handle = thread::Builder::new()
            .name(format!(
                "Runtime {}, cuda={}, CpuSet={}",
                runtime_id.0, cuda_dev, cores
            ))
            .spawn(move || {
                if !cores.sched_set_affinity_for_current_thread() {
                    log::warn!("Set affinity for {:?} failed", runtime_id);
                }
                log::info!(
                    "Runtime {} started, cuda={}, CpuSet={}",
                    runtime_id.0,
                    cuda_dev,
                    cores
                );
                runtime_thread.mainloop();
            })
            .unwrap();
        RuntimeHandle {
            id: runtime_id,
            runtime,
            join_handle,
        }
    }
}
