use burn::{
  nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
  prelude::*,
  tensor::activation::leaky_relu,
};

use super::{ConvBottom, ConvBottomConfig, UNetConv, UNetConvConfig};

#[derive(Debug, Module)]
pub struct UNet1<B: Backend> {
  conv1: UNetConv<B>,
  conv1_down: Conv2d<B>,
  conv2: UNetConv<B>,
  conv2_up: ConvTranspose2d<B>,
  conv3: Conv2d<B>,
  conv_bottom: ConvBottom<B>,
}

impl<B: Backend> UNet1<B> {
  pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
    let x1 = self.conv1.forward(x);
    let x2 = self.conv1_down.forward(x1.clone());

    let x1 = {
      let [_, _, h, w] = x1.dims();
      x1.narrow(3, 4, w - 8).narrow(2, 4, h - 8)
    };

    let x2 = leaky_relu(x2, 0.1);
    let x2 = self.conv2.forward(x2);
    let x2 = self.conv2_up.forward(x2);
    let x2 = leaky_relu(x2, 0.1);

    let x3 = self.conv3.forward(x1 + x2);
    let x3 = leaky_relu(x3, 0.1);

    self.conv_bottom.forward(x3)
  }
}

#[derive(Config)]
pub struct UNet1Config {
  in_channels: usize,
  out_channels: usize,
  deconv: bool,
}

impl UNet1Config {
  pub fn init<B: Backend>(&self, device: &B::Device) -> UNet1<B> {
    UNet1 {
      conv1: UNetConvConfig::new(self.in_channels, 32, 64, false).init(device),
      conv1_down: Conv2dConfig::new([64, 64], [2, 2])
        .with_stride([2, 2])
        .init(device),
      conv2: UNetConvConfig::new(64, 128, 64, true).init(device),
      conv2_up: ConvTranspose2dConfig::new([64, 64], [2, 2])
        .with_stride([2, 2])
        .init(device),
      conv3: Conv2dConfig::new([64, 64], [3, 3]).init(device),
      conv_bottom: ConvBottomConfig::new(self.out_channels, self.deconv).init(device),
    }
  }
}
