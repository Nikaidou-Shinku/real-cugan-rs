mod se_block;
mod unet1;
mod unet2;
mod unet_conv;

use candle_core::{Module, Tensor};
use candle_nn::{Conv2d, ConvTranspose2d};

pub use se_block::*;
pub use unet1::*;
pub use unet2::*;
pub use unet_conv::*;

enum ConvBottom {
  Deconv(ConvTranspose2d),
  Else(Conv2d),
}

impl Module for ConvBottom {
  fn forward(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
    match self {
      ConvBottom::Deconv(m) => m.forward(x),
      ConvBottom::Else(m) => m.forward(x),
    }
  }
}
