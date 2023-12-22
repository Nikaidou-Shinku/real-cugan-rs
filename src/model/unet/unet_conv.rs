use candle_core::{Module, Tensor};
use candle_nn::{conv2d, seq, Activation, Conv2dConfig, Sequential, VarBuilder};

use super::SeBlock;

pub struct UNetConv {
  pub conv: Sequential,
  pub seblock: Option<SeBlock>,
}

impl UNetConv {
  pub fn new(
    in_channels: usize,
    mid_channels: usize,
    out_channels: usize,
    se: bool,
    vb: VarBuilder,
  ) -> Result<Self, candle_core::Error> {
    let mut conv = seq();

    conv = conv.add(conv2d(
      in_channels,
      mid_channels,
      3,
      Conv2dConfig::default(),
      vb.pp("conv.0"),
    )?);

    conv = conv.add(Activation::LeakyRelu(0.1));

    conv = conv.add(conv2d(
      mid_channels,
      out_channels,
      3,
      Conv2dConfig::default(),
      vb.pp("conv.2"),
    )?);

    conv = conv.add(Activation::LeakyRelu(0.1));

    let seblock = if se {
      Some(SeBlock::new(out_channels, 8, true, vb.pp("seblock"))?)
    } else {
      None
    };

    Ok(Self { conv, seblock })
  }
}

impl Module for UNetConv {
  fn forward(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
    let mut z = self.conv.forward(x)?;

    if let Some(seblock) = &self.seblock {
      z = seblock.forward(&z)?;
    }

    Ok(z)
  }
}
