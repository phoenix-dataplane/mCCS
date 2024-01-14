pub enum SchedulingMode {
    Dedicated,
    Compact,
}

pub enum EngineStatus {
    Idle,
    Progressed,
    Completed,
}

pub trait Engine: Send + Unpin + 'static {
    fn progress(&mut self) -> EngineStatus;

    fn scheduling_hint(&self) -> SchedulingMode {
        SchedulingMode::Dedicated
    }
}
