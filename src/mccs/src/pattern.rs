pub struct RingPattern {
    pub prev: usize,
    pub next: usize,
    pub user_ranks: Vec<usize>,
    // rank 0's distance to my rank along the ring send path
    pub index: usize,
}
