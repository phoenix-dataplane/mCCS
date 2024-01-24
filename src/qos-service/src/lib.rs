use std::collections::HashMap;

use serde::{Serialize, Deserialize};
use interval::IntervalSet;
use interval::interval_set::ToIntervalSet;


#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(transparent)]
#[serde(transparent)]
pub struct CommunicatorId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum QosMode {
    Allow,
    Deny,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QosIntervalDef {
    // start and end timestamps in microseconds
    pub intervals: Vec<(u64, u64)>,
    pub mode: QosMode,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QosScheduleDef {
    #[serde(deserialize_with = "deserialize_schedule")]
    pub schedule: HashMap<CommunicatorId, QosIntervalDef>, 
    pub epoch_microsecs: u64,
}

fn deserialize_schedule<'de, D>(deserializer: D) -> Result<HashMap<CommunicatorId, QosIntervalDef>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let map: HashMap<String, QosIntervalDef> = Deserialize::deserialize(deserializer)?;
    map.into_iter()
       .map(|(k, v)| 
            k.parse::<u32>().map_err(serde::de::Error::custom)
            .map(|key| (CommunicatorId(key), v))
        ).collect()
}

#[derive(Clone, Debug)]
pub struct QosInterval {
    pub intervals: IntervalSet<u64>,
    pub mode: QosMode
}

impl From<QosIntervalDef> for QosInterval {
    fn from(def: QosIntervalDef) -> Self {
        let intervals = def.intervals.to_interval_set();
        QosInterval {
            intervals,
            mode: def.mode
        }
    }
}

#[derive(Clone, Debug)]
pub struct QosSchedule {
    pub schedule: HashMap<CommunicatorId, QosInterval>,
    pub epoch_microsecs: u64,
}

impl From<QosScheduleDef> for QosSchedule {
    fn from(def: QosScheduleDef) -> Self {
        let schedule = def.schedule
        .into_iter()
        .map(|(k, v)| (k, v.into())).collect();
        QosSchedule {
            schedule,
            epoch_microsecs: def.epoch_microsecs,
        }
    }
}