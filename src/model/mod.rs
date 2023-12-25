mod unet;
mod up_cunet;

use candle_core::{Module, Tensor};

pub use up_cunet::*;

pub enum RealCugan {
  X2(UpCunet2x),
  X3(UpCunet3x),
}

impl Module for RealCugan {
  fn forward(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
    match self {
      RealCugan::X2(m) => m.forward(x),
      RealCugan::X3(m) => m.forward(x),
    }
  }
}
