use crate::comm::Communicator;

use super::init::CommSuspendState;

pub enum ReconfigState {
    Reserved,
    Start(Communicator, CommSuspendState),
    UntilLaunchNTasks(u64, Communicator, CommSuspendState),
    Ready(Communicator, CommSuspendState),
}

impl ReconfigState {
    pub fn start(communicator: Communicator, suspend_state: CommSuspendState) -> Self {
        ReconfigState::Start(communicator, suspend_state)
    }

    pub fn update_val(&mut self, val: u64) {
        let rhs = std::mem::replace(self, ReconfigState::Reserved);
        *self = match rhs {
            ReconfigState::Start(comm, suspend_state) => {
                ReconfigState::UntilLaunchNTasks(val, comm, suspend_state)
            }
            _ => unreachable!(),
        }
    }

    pub fn set_ready(&mut self) {
        let rhs = std::mem::replace(self, ReconfigState::Reserved);
        *self = match rhs {
            ReconfigState::UntilLaunchNTasks(val, comm, suspend_state) => {
                ReconfigState::Ready(comm, suspend_state)
            }
            _ => unreachable!(),
        }
    }

    pub fn to_inner(self) -> (Communicator, CommSuspendState) {
        match self {
            ReconfigState::Ready(comm, suspend_state) => (comm, suspend_state),
            _ => unreachable!(),
        }
    }
}
