use candle_core::{Module, Tensor};
use candle_nn::VarBuilder;

use crate::model::unet::{UNet1, UNet2};

pub struct UpCunet2x {
  unet1: UNet1,
  unet2: UNet2,
}

impl UpCunet2x {
  pub fn new(
    vb: VarBuilder,
    in_channels: usize,
    out_channels: usize,
  ) -> Result<Self, candle_core::Error> {
    let unet1 = UNet1::new(vb.pp("unet1"), in_channels, out_channels, true)?;
    let unet2 = UNet2::new(vb.pp("unet2"), in_channels, out_channels, false)?;

    Ok(Self { unet1, unet2 })
  }
}

impl Module for UpCunet2x {
  fn forward(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
    let (_, _, h0, w0) = x.shape().dims4()?;

    let ph = ((h0 - 1) / 2 + 1) * 2;
    let pw = ((w0 - 1) / 2 + 1) * 2;

    // TODO: reflection pad
    let mut x = x
      .pad_with_zeros(3, 18, 18 + pw - w0)?
      .pad_with_zeros(2, 18, 18 + ph - h0)?;

    x = self.unet1.forward(&x)?;

    let x0 = self.unet2.forward(&x)?;

    x = x
      .narrow(3, 20, x.dim(3)? - 40)?
      .narrow(2, 20, x.dim(2)? - 40)?;

    x = x0.add(&x)?;

    ((x - 0.15)? * (255. / 0.7))?.round()
  }
}
