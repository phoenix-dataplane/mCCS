pub struct WorkPool<T> {
    pool: Vec<T>,
}

impl<T> WorkPool<T> {
    pub fn new() -> Self {
        WorkPool { pool: Vec::new() }
    }
}

impl<T> Default for WorkPool<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> WorkPool<T> {
    pub fn progress<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut T) -> bool,
    {
        let mut idx = 0;
        while idx < self.pool.len() {
            let finished = f(&mut self.pool[idx]);
            if finished {
                self.pool.swap_remove(idx);
            } else {
                idx += 1;
            }
        }
    }

    pub fn enqueue(&mut self, elem: T) {
        self.pool.push(elem);
    }
}
