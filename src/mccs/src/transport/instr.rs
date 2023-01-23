// TODO: align with ncclProxyArgs and ncclProxySubArgs
#[allow(unused)]
struct TransportInstruction {
    communicator_id: u64,
    n_steps: u64,
    curr_step: u64,
}