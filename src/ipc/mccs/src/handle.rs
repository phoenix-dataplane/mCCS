use std::os::raw::c_char;

use serde::{Serialize, Deserialize};
use serde_big_array::BigArray;

#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaMemHandle(#[serde(with = "BigArray")] pub [c_char; 64usize]);