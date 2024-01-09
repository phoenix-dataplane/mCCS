use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};

use super::op::TransportOp;
use super::transporter::TransportAgentId;

const PER_CONN_QUEUE_INIT_CAPACITY: usize = 16;

pub struct TransrportOpQueue {
    queue: Vec<(TransportAgentId, VecDeque<TransportOp>)>,
    active_connections: usize,
    connections_index_map: HashMap<TransportAgentId, usize>,
}

impl TransrportOpQueue {
    pub fn new() -> Self {
        TransrportOpQueue {
            queue: Vec::new(),
            active_connections: 0,
            connections_index_map: HashMap::new(),
        }
    }

    pub fn submit_op(&mut self, agent: TransportAgentId, op: TransportOp) {
        match self.connections_index_map.entry(agent) {
            Entry::Occupied(entry) => {
                let index = *entry.get();
                self.queue[index].1.push_back(op);
            }
            Entry::Vacant(entry) => {
                if self.active_connections < self.queue.len() {
                    debug_assert!(self.queue[self.active_connections].1.is_empty());
                    self.queue[self.active_connections].0 = agent;
                    self.queue[self.active_connections].1.push_back(op);
                } else {
                    self.queue
                        .push((agent, VecDeque::with_capacity(PER_CONN_QUEUE_INIT_CAPACITY)));
                }
                entry.insert(self.active_connections);
                self.active_connections += 1;
            }
        }
    }

    pub fn progress_ops<F>(&mut self, mut f: F)
    where
        F: FnMut(&TransportAgentId, &mut TransportOp) -> bool,
    {
        let mut conn_idx = 0;
        while conn_idx < self.active_connections {
            let (agent, conn_queue) = &mut self.queue[conn_idx];
            let finished = f(agent, &mut conn_queue[0]);
            let mut conn_inc = 1;
            if finished {
                conn_queue.pop_front();
                if conn_queue.is_empty() {
                    self.connections_index_map.remove(agent);
                    self.queue.swap_remove(conn_idx);
                    self.active_connections -= 1;
                    conn_inc = 0;
                }
            }
            conn_idx += conn_inc;
        }
    }
}
