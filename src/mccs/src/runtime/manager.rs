use std::sync::Arc;
use std::sync::Mutex;
use std::thread::{self, JoinHandle};

use super::{EngineContainer, Runtime, RuntimeMode};
use crate::engine::SchedulingMode;

struct RuntimeHandle {
    runtime: Arc<Runtime>,
    join_handle: JoinHandle<()>,
}

pub struct RuntimeManager {
    runtimes: Mutex<Vec<RuntimeHandle>>,
}

impl RuntimeManager {
    pub fn new() -> Self {
        RuntimeManager {
            runtimes: Mutex::new(Vec::new()),
        }
    }

    pub fn submit_engine(&self, engine: EngineContainer, cuda_dev: Option<i32>) {
        let mode = engine.scheduling_hint();
        let runtime_mode = match mode {
            SchedulingMode::Dedicated => RuntimeMode::Dedicated,
            SchedulingMode::Compact => RuntimeMode::Shared,
        };
        let mut runtimes = self.runtimes.lock().unwrap();
        let r = runtimes
            .iter()
            .find(|r| r.runtime.try_acquire(runtime_mode, cuda_dev, None));
        if let Some(handle) = r {
            handle.runtime.add_engine(engine);
            handle.join_handle.thread().unpark();
        } else {
            let handle = Self::start_runtime(cuda_dev, runtime_mode);
            handle.runtime.add_engine(engine);
            handle.join_handle.thread().unpark();
            runtimes.push(handle);
        }
    }

    fn start_runtime(cuda_dev: Option<i32>, mode: RuntimeMode) -> RuntimeHandle {
        let runtime = Arc::new(Runtime::new(cuda_dev, mode));
        let flag = runtime.try_acquire(mode, cuda_dev, None);
        assert!(flag);

        let cuda_dev = cuda_dev.unwrap_or(-1);
        let runtime_thread = Arc::clone(&runtime);
        let join_handle = thread::Builder::new()
            .name(format!("Runtime-cuda:{}", cuda_dev))
            .spawn(move || {
                runtime_thread.mainloop();
            })
            .unwrap();
        RuntimeHandle {
            runtime,
            join_handle,
        }
    }
}
