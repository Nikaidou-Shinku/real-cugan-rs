use candle_core::{Module, Tensor};
use candle_nn::{
  conv2d, conv_transpose2d, ops::leaky_relu, Conv2d, Conv2dConfig, ConvTranspose2d,
  ConvTranspose2dConfig, VarBuilder,
};

use super::{ConvBottom, UNetConv};

pub struct UNet1 {
  conv1: UNetConv,
  conv1_down: Conv2d,
  pub conv2: UNetConv,
  conv2_up: ConvTranspose2d,
  conv3: Conv2d,
  conv_bottom: ConvBottom,
}

impl UNet1 {
  pub fn new(
    in_channels: usize,
    out_channels: usize,
    deconv: bool,
    for_x3: bool,
    vb: VarBuilder,
  ) -> Result<Self, candle_core::Error> {
    let conv1 = UNetConv::new(in_channels, 32, 64, false, vb.pp("conv1"))?;
    let conv1_down = conv2d(
      64,
      64,
      2,
      Conv2dConfig {
        padding: 0,
        stride: 2,
        dilation: 1,
        groups: 1,
      },
      vb.pp("conv1_down"),
    )?;

    let conv2 = UNetConv::new(64, 128, 64, true, vb.pp("conv2"))?;
    let conv2_up = conv_transpose2d(
      64,
      64,
      2,
      ConvTranspose2dConfig {
        padding: 0,
        output_padding: 0,
        stride: 2,
        dilation: 1,
      },
      vb.pp("conv2_up"),
    )?;

    let conv3 = conv2d(64, 64, 3, Conv2dConfig::default(), vb.pp("conv3"))?;

    let conv_bottom = if deconv {
      ConvBottom::Deconv(conv_transpose2d(
        64,
        out_channels,
        if for_x3 { 5 } else { 4 },
        ConvTranspose2dConfig {
          padding: if for_x3 { 2 } else { 3 },
          output_padding: 0,
          stride: if for_x3 { 3 } else { 2 },
          dilation: 1,
        },
        vb.pp("conv_bottom"),
      )?)
    } else {
      ConvBottom::Else(conv2d(
        64,
        out_channels,
        3,
        Conv2dConfig::default(),
        vb.pp("conv_bottom"),
      )?)
    };

    Ok(Self {
      conv1,
      conv1_down,
      conv2,
      conv2_up,
      conv3,
      conv_bottom,
    })
  }
}

impl Module for UNet1 {
  fn forward(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
    let mut x1 = self.conv1.forward(x)?;
    let mut x2 = self.conv1_down.forward(&x1)?;

    x1 = x1
      .narrow(3, 4, x1.dim(3)? - 8)?
      .narrow(2, 4, x1.dim(2)? - 8)?;

    x2 = leaky_relu(&x2, 0.1)?;
    x2 = self.conv2.forward(&x2)?;
    x2 = self.conv2_up.forward(&x2)?;
    x2 = leaky_relu(&x2, 0.1)?;

    let mut x3 = self.conv3.forward(&(x1 + x2)?)?;
    x3 = leaky_relu(&x3, 0.1)?;

    self.conv_bottom.forward(&x3)
  }
}

impl UNet1 {
  pub fn forward_a(&self, x: &Tensor) -> Result<(Tensor, Tensor), candle_core::Error> {
    let mut x1 = self.conv1.forward(x)?;
    let mut x2 = self.conv1_down.forward(&x1)?;

    x1 = x1
      .narrow(3, 4, x1.dim(3)? - 8)?
      .narrow(2, 4, x1.dim(2)? - 8)?;

    x2 = leaky_relu(&x2, 0.1)?;
    x2 = self.conv2.conv.forward(&x2)?;

    Ok((x1, x2))
  }

  pub fn forward_b(&self, x1: &Tensor, x2: &Tensor) -> Result<Tensor, candle_core::Error> {
    let mut x2 = self.conv2_up.forward(x2)?;
    x2 = leaky_relu(&x2, 0.1)?;

    let mut x3 = self.conv3.forward(&(x1 + x2)?)?;
    x3 = leaky_relu(&x3, 0.1)?;

    self.conv_bottom.forward(&x3)
  }
}
