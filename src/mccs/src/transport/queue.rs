use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};

use super::op::TransportOp;
use super::transporter::TransportAgentId;

const PER_CONN_QUEUE_INIT_CAPACITY: usize = 16;

pub struct TransrportOpQueue {
    // bool: mark for delayed removal
    queue: Vec<(TransportAgentId, VecDeque<TransportOp>, bool)>,
    connections_index_map: HashMap<TransportAgentId, usize>,
    // agents that have completed all outstanding ops and are removed
    // transport engine needs to remove corresponding resources
    removed_agents: Vec<TransportAgentId>,
    // indices of transport agents in `queue` that are marked for removal
    removal_indices: Vec<usize>,
}

impl TransrportOpQueue {
    pub fn new() -> Self {
        TransrportOpQueue {
            queue: Vec::new(),
            connections_index_map: HashMap::new(),
            removed_agents: Vec::new(),
            removal_indices: Vec::new(),
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
                entry.insert(idx);
                self.queue.push((agent, agent_queue, false));
            }
        }
    }

    pub fn progress_ops<F>(&mut self, mut f: F) -> &mut Vec<TransportAgentId>
    where
        F: FnMut(&TransportAgentId, &mut TransportOp) -> bool,
    {
        for (idx, (agent_id, agent_queue, removal)) in self.queue.iter_mut().enumerate() {
            if !agent_queue.is_empty() {
                let finished = f(agent_id, &mut agent_queue[0]);
                if finished {
                    agent_queue.pop_front();
                }
            } else {
                if *removal {
                    self.removed_agents.push(*agent_id);
                    self.removal_indices.push(idx);
                    self.connections_index_map.remove(agent_id);
                }
            }
        }

        for mut idx in self.removal_indices.drain(..).rev() {
            self.queue.swap_remove(idx);
            if self.queue.len() > 0 {
                if idx < self.queue.len() {
                    let agent_id = self.queue[idx].0;
                    self.connections_index_map.insert(agent_id, idx);
                }
            }
        }
        &mut self.removed_agents
    }

    // Returns true if there is no oustanding ops for that agent,
    // otherwise returns false and marks the agent for delayed removal
    pub fn remove_agent(&mut self, agent_id: &TransportAgentId) -> bool {
        if let Some(mut index) = self.connections_index_map.get(agent_id).copied() {
            let (agent_id, agent_queue, removal) = &mut self.queue[index];
            if agent_queue.is_empty() {
                let agent_id = *agent_id;
                self.queue.swap_remove(index);
                if self.queue.len() > 0 {
                    if index < self.queue.len() {
                        let swap_agent_id = self.queue[index].0;
                        self.connections_index_map.insert(swap_agent_id, index);
                    }
                }
                self.connections_index_map.remove(&agent_id);
                true
            } else {
                *removal = true;
                false
            }
        } else {
            true
        }
    }
}
