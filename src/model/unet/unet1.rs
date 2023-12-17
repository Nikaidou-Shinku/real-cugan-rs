use candle_nn::{
  conv2d, conv_transpose2d, Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig,
  VarBuilder,
};

use super::UNetConv;

enum UNet1ConvBottom {
  Deconv(ConvTranspose2d),
  Else(Conv2d),
}

pub struct UNet1 {
  conv1: UNetConv,
  conv1_down: Conv2d,
  conv2: UNetConv,
  conv2_up: ConvTranspose2d,
  conv3: Conv2d,
  conv_bottom: UNet1ConvBottom,
}

impl UNet1 {
  pub fn new(
    vb: &VarBuilder,
    in_channels: usize,
    out_channels: usize,
    deconv: bool,
  ) -> Result<Self, candle_core::Error> {
    let conv1 = UNetConv::new(&vb.pp("unet1_conv1"), in_channels, 32, 64, false)?;
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
      vb.pp("unet1_conv1_down"),
    )?;

    let conv2 = UNetConv::new(&vb.pp("unet1_conv2"), 64, 128, 64, true)?;
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
      vb.pp("unet1_conv2_up"),
    )?;

    let conv3 = conv2d(64, 64, 3, Conv2dConfig::default(), vb.pp("unet1_conv3"))?;

    let conv_bottom = if deconv {
      UNet1ConvBottom::Deconv(conv_transpose2d(
        64,
        out_channels,
        4,
        ConvTranspose2dConfig {
          padding: 3,
          output_padding: 0,
          stride: 2,
          dilation: 1,
        },
        vb.pp("unet1_conv_bottom"),
      )?)
    } else {
      UNet1ConvBottom::Else(conv2d(
        64,
        out_channels,
        3,
        Conv2dConfig::default(),
        vb.pp("unet1_conv_bottom"),
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
