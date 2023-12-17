use candle_core::{Module, Tensor};
use candle_nn::{conv2d, conv2d_no_bias, ops::sigmoid, Conv2d, Conv2dConfig, VarBuilder};

pub struct SeBlock {
  conv1: Conv2d,
  conv2: Conv2d,
}

impl SeBlock {
  pub fn new(
    vb: VarBuilder,
    in_channels: usize,
    reduction: usize,
    bias: bool,
  ) -> Result<Self, candle_core::Error> {
    let conv1 = if bias { conv2d } else { conv2d_no_bias }(
      in_channels,
      in_channels / reduction,
      1,
      Conv2dConfig::default(),
      vb.pp("conv1"),
    )?;

    let conv2 = if bias { conv2d } else { conv2d_no_bias }(
      in_channels / reduction,
      in_channels,
      1,
      Conv2dConfig::default(),
      vb.pp("conv2"),
    )?;

    Ok(Self { conv1, conv2 })
  }
}

impl Module for SeBlock {
  fn forward(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
    let mut x0 = x.mean_keepdim((2, 3))?;
    x0 = self.conv1.forward(&x0)?;
    x0 = x0.relu()?;
    x0 = self.conv2.forward(&x0)?;
    x0 = sigmoid(&x0)?;
    x.broadcast_mul(&x0)
  }
}
