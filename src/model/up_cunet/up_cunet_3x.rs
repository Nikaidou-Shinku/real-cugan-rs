use candle_core::{Module, Tensor};
use candle_nn::VarBuilder;

use crate::{
  model::unet::{UNet1, UNet2},
  utils::TensorExt,
};

pub struct UpCunet3x {
  unet1: UNet1,
  unet2: UNet2,
  // alpha: f64,
  tile_size: Option<usize>,
  // use_cache: bool,
}

impl UpCunet3x {
  pub fn new(
    in_channels: usize,
    out_channels: usize,
    alpha: f64,
    tile_size: Option<usize>,
    // use_cache: bool,
    vb: VarBuilder,
  ) -> Result<Self, candle_core::Error> {
    let unet1 = UNet1::new(in_channels, out_channels, true, true, vb.pp("unet1"))?;
    let unet2 = UNet2::new(in_channels, out_channels, false, alpha, vb.pp("unet2"))?;

    Ok(Self {
      unet1,
      unet2,
      tile_size,
    })
  }
}

impl Module for UpCunet3x {
  fn forward(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
    if self.tile_size.is_some() {
      todo!("3x upscale with tile mode is not supported for now!");
    }

    let (_, _, h0, w0) = x.shape().dims4()?;

    let ph = ((h0 - 1) / 4 + 1) * 4;
    let pw = ((w0 - 1) / 4 + 1) * 4;

    let mut x = x
      .reflection_pad(3, 14, 14 + pw - w0)?
      .reflection_pad(2, 14, 14 + ph - h0)?;

    x = self.unet1.forward(&x)?;

    let x0 = self.unet2.forward(&x)?;

    x = x
      .narrow(3, 20, x.dim(3)? - 40)?
      .narrow(2, 20, x.dim(2)? - 40)?;

    x = x0.add(&x)?;

    if w0 != pw || h0 != ph {
      x = x.narrow(3, 0, w0 * 3)?.narrow(2, 0, h0 * 3)?;
    }

    Ok(x)
  }
}
