pub const MCCS_STEP: u32 = 8;
pub const ALLGATHER_CHUNK_STEPS: u32 = MCCS_STEP / 2;
pub const ALLREDUCE_CHUNK_STEPS: u32 = MCCS_STEP / 2;
pub const ALLGATHER_SLICE_STEPS: u32 = MCCS_STEP / 4;
pub const ALLREDUCE_SLICE_STEPS: u32 = MCCS_STEP / 4;

#[derive(Clone, Debug)]
pub struct RingPattern {
    pub prev: usize,
    pub next: usize,
    pub user_ranks: Vec<usize>,
    // rank 0's distance to my rank along the ring send path
    pub index: usize,
}
