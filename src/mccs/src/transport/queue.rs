use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};

use super::op::TransportOp;
use super::transporter::TransportAgentId;

const PER_CONN_QUEUE_INIT_CAPACITY: usize = 16;

pub struct TransrportOpQueue {
    queue: Vec<(TransportAgentId, VecDeque<TransportOp>)>,
    connections_index_map: HashMap<TransportAgentId, usize>,
}

impl TransrportOpQueue {
    pub fn new() -> Self {
        TransrportOpQueue {
            queue: Vec::new(),
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
                let mut agent_queue = VecDeque::with_capacity(PER_CONN_QUEUE_INIT_CAPACITY);
                agent_queue.push_back(op);
                let idx = self.queue.len();
                self.queue.push((agent, agent_queue));
                entry.insert(idx);
            }
        }
    }

    pub fn progress_ops<F>(&mut self, mut f: F)
    where
        F: FnMut(&TransportAgentId, &mut TransportOp) -> bool,
    {
        for (agent_id, agent_queue) in self.queue.iter_mut() {
            if !agent_queue.is_empty() {
                let finished = f(agent_id, &mut agent_queue[0]);
                if finished {
                    agent_queue.pop_front();
                }
            }
        }
    }

    pub fn remove_agent(&mut self, agent_id: &TransportAgentId) {
        // TODO
        // if let Some(index) = self.connections_index_map.remove(agent_id) {
        //     self.queue.swap_remove(index);
        // }
    }
}
