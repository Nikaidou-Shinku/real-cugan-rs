use candle_core::{Module, Tensor};
use candle_nn::{
  conv2d, conv_transpose2d, ops::leaky_relu, Conv2d, Conv2dConfig, ConvTranspose2d,
  ConvTranspose2dConfig, VarBuilder,
};

use super::{ConvBottom, UNetConv};

pub struct UNet2 {
  conv1: UNetConv,
  conv1_down: Conv2d,
  conv2: UNetConv,
  conv2_down: Conv2d,
  conv3: UNetConv,
  conv3_up: ConvTranspose2d,
  conv4: UNetConv,
  conv4_up: ConvTranspose2d,
  conv5: Conv2d,
  conv_bottom: ConvBottom,
}

impl UNet2 {
  pub fn new(
    in_channels: usize,
    out_channels: usize,
    deconv: bool,
    vb: VarBuilder,
  ) -> Result<Self, candle_core::Error> {
    let conf1 = Conv2dConfig {
      padding: 0,
      stride: 2,
      dilation: 1,
      groups: 1,
    };
    let conf2 = ConvTranspose2dConfig {
      padding: 0,
      output_padding: 0,
      stride: 2,
      dilation: 1,
    };

    let conv1 = UNetConv::new(in_channels, 32, 64, false, vb.pp("conv1"))?;
    let conv1_down = conv2d(64, 64, 2, conf1, vb.pp("conv1_down"))?;

    let conv2 = UNetConv::new(64, 64, 128, true, vb.pp("conv2"))?;
    let conv2_down = conv2d(128, 128, 2, conf1, vb.pp("conv2_down"))?;

    let conv3 = UNetConv::new(128, 256, 128, true, vb.pp("conv3"))?;
    let conv3_up = conv_transpose2d(128, 128, 2, conf2, vb.pp("conv3_up"))?;

    let conv4 = UNetConv::new(128, 64, 64, true, vb.pp("conv4"))?;
    let conv4_up = conv_transpose2d(64, 64, 2, conf2, vb.pp("conv4_up"))?;

    let conv5 = conv2d(64, 64, 3, Conv2dConfig::default(), vb.pp("conv5"))?;

    let conv_bottom = if deconv {
      ConvBottom::Deconv(conv_transpose2d(
        64,
        out_channels,
        4,
        ConvTranspose2dConfig {
          padding: 3,
          output_padding: 0,
          stride: 2,
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
      conv2_down,
      conv3,
      conv3_up,
      conv4,
      conv4_up,
      conv5,
      conv_bottom,
    })
  }
}

impl Module for UNet2 {
  fn forward(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
    let mut x1 = self.conv1.forward(x)?;
    let mut x2 = self.conv1_down.forward(&x1)?;

    x1 = x1
      .narrow(3, 16, x1.dim(3)? - 32)?
      .narrow(2, 16, x1.dim(2)? - 32)?;

    x2 = leaky_relu(&x2, 0.1)?;
    x2 = self.conv2.forward(&x2)?;

    let mut x3 = self.conv2_down.forward(&x2)?;

    x2 = x2
      .narrow(3, 4, x2.dim(3)? - 8)?
      .narrow(2, 4, x2.dim(2)? - 8)?;

    x3 = leaky_relu(&x3, 0.1)?;
    x3 = self.conv3.forward(&x3)?;
    x3 = self.conv3_up.forward(&x3)?;
    x3 = leaky_relu(&x3, 0.1)?;

    let mut x4 = self.conv4.forward(&(x2 + x3)?)?;
    // TODO: alpha?
    x4 = self.conv4_up.forward(&x4)?;
    x4 = leaky_relu(&x4, 0.1)?;

    let mut x5 = self.conv5.forward(&(x1 + x4)?)?;
    x5 = leaky_relu(&x5, 0.1)?;

    self.conv_bottom.forward(&x5)
  }
}
