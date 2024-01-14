use crossbeam::channel::Sender;
use dashmap::DashMap;

use super::engine::TransportEngineId;
use super::transporter::TransportAgentId;
use crate::message::ControlRequest;

const MAX_CONNS_PER_ENGINE: usize = 8;
const MAX_NUM_ENGINES_PER_DEVICE: usize = 8;

pub struct TransportDelegator {
    // cuda dev -> Vec<(engine_idx, num_agents)>
    pub active_connections: DashMap<i32, Vec<(u32, usize)>>,
    pub shutdown_engines: DashMap<i32, Vec<u32>>,
}

impl TransportDelegator {
    pub fn new() -> Self {
        TransportDelegator {
            active_connections: DashMap::new(),
            shutdown_engines: DashMap::new(),
        }
    }
}

impl Default for TransportDelegator {
    fn default() -> Self {
        Self::new()
    }
}

impl TransportDelegator {
    pub fn assign_transport_engine(
        &self,
        cuda_dev: i32,
        agent: TransportAgentId,
        control: &mut Sender<ControlRequest>,
    ) -> TransportEngineId {
        let mut engines = self
            .active_connections
            .entry(cuda_dev)
            .or_insert_with(Vec::new);
        let num_engines = engines.len();

        let least_load = engines.iter_mut().min_by_key(|x| x.1);

        if let Some((engine_idx, conns)) = least_load {
            if (*conns < MAX_CONNS_PER_ENGINE) || (num_engines >= MAX_NUM_ENGINES_PER_DEVICE) {
                let engine = TransportEngineId {
                    cuda_device_idx: cuda_dev,
                    index: *engine_idx,
                };
                *conns += 1;
                return engine;
            }
        }
        let reusable_indices = self.shutdown_engines.get_mut(&cuda_dev);
        let idx = if let Some(mut indices) = reusable_indices {
            indices.pop()
        } else {
            None
        };
        let idx = idx.unwrap_or_else(|| engines.iter().max_by_key(|x| x.0).map_or(0, |x| x.0 + 1));
        engines.push((idx, 1));
        let new_engine = TransportEngineId {
            cuda_device_idx: cuda_dev,
            index: idx,
        };
        control
            .send(ControlRequest::NewTransportEngine(new_engine))
            .unwrap();
        new_engine
    }

    pub fn register_agent_shutdown(&self, engine: TransportEngineId) {
        let mut engines = self
            .active_connections
            .entry(engine.cuda_device_idx)
            .or_insert_with(Vec::new);
        let engine_idx = engines.iter().position(|x| x.0 == engine.index).unwrap();
        engines[engine_idx].1 -= 1;
    }

    pub fn register_engine_shutdown(&self, engine: TransportEngineId) {
        let mut engines = self
            .active_connections
            .entry(engine.cuda_device_idx)
            .or_insert_with(Vec::new);
        let engine_idx = engines.iter().position(|x| x.0 == engine.index).unwrap();
        engines.remove(engine_idx);
        let mut shutdown_engines = self
            .shutdown_engines
            .entry(engine.cuda_device_idx)
            .or_insert_with(Vec::new);
        shutdown_engines.push(engine.index);
    }
}
