pub struct DuplexChannel<T, R> {
    pub tx: crossbeam::channel::Sender<T>,
    pub rx: crossbeam::channel::Receiver<R>,
}

impl<T, R> DuplexChannel<T, R> {
    pub fn new_unbound_pair() -> (DuplexChannel<T, R>, DuplexChannel<R, T>) {
        let (t_tx, t_rx) = crossbeam::channel::unbounded();
        let (r_tx, r_rx) = crossbeam::channel::unbounded();
        (
            DuplexChannel { tx: t_tx, rx: r_rx },
            DuplexChannel { tx: r_tx, rx: t_rx },
        )
    }

    pub fn new_bound_pair(
        t_to_r: usize,
        r_to_t: usize,
    ) -> (DuplexChannel<T, R>, DuplexChannel<R, T>) {
        let (t_tx, t_rx) = crossbeam::channel::bounded(t_to_r);
        let (r_tx, r_rx) = crossbeam::channel::bounded(r_to_t);
        (
            DuplexChannel { tx: t_tx, rx: r_rx },
            DuplexChannel { tx: r_tx, rx: t_rx },
        )
    }
}
