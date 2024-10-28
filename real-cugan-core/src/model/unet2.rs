use burn::{
  nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
  prelude::*,
  tensor::activation::leaky_relu,
};

use super::{ConvBottom, ConvBottomConfig, UNetConv, UNetConvConfig};

#[derive(Debug, Module)]
pub struct UNet2<B: Backend> {
  conv1: UNetConv<B>,
  conv1_down: Conv2d<B>,
  conv2: UNetConv<B>,
  conv2_down: Conv2d<B>,
  conv3: UNetConv<B>,
  conv3_up: ConvTranspose2d<B>,
  conv4: UNetConv<B>,
  conv4_up: ConvTranspose2d<B>,
  conv5: Conv2d<B>,
  conv_bottom: ConvBottom<B>,
  alpha: f64,
}

impl<B: Backend> UNet2<B> {
  pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
    let x1 = self.conv1.forward(x);
    let x2 = self.conv1_down.forward(x1.clone());

    let x1 = {
      let [_, _, h, w] = x1.dims();
      x1.narrow(3, 16, w - 32).narrow(2, 16, h - 32)
    };

    let x2 = leaky_relu(x2, 0.1);
    let x2 = self.conv2.forward(x2);
    let x3 = self.conv2_down.forward(x2.clone());

    let x2 = {
      let [_, _, h, w] = x2.dims();
      x2.narrow(3, 4, w - 8).narrow(2, 4, h - 8)
    };

    let x3 = leaky_relu(x3, 0.1);
    let x3 = self.conv3.forward(x3);
    let x3 = self.conv3_up.forward(x3);
    let x3 = leaky_relu(x3, 0.1);

    let x4 = self.conv4.forward(x2 + x3);
    let x4 = x4.mul_scalar(self.alpha);
    let x4 = self.conv4_up.forward(x4);
    let x4 = leaky_relu(x4, 0.1);

    let x5 = self.conv5.forward(x1 + x4);
    let x5 = leaky_relu(x5, 0.1);

    self.conv_bottom.forward(x5)
  }
}

#[derive(Config)]
pub struct UNet2Config {
  in_channels: usize,
  out_channels: usize,
  deconv: bool,
  #[config(default = 1.0)]
  alpha: f64,
}

impl UNet2Config {
  pub fn init<B: Backend>(&self, device: &B::Device) -> UNet2<B> {
    UNet2 {
      conv1: UNetConvConfig::new(self.in_channels, 32, 64, false).init(device),
      conv1_down: Conv2dConfig::new([64, 64], [2, 2])
        .with_stride([2, 2])
        .init(device),
      conv2: UNetConvConfig::new(64, 64, 128, true).init(device),
      conv2_down: Conv2dConfig::new([128, 128], [2, 2])
        .with_stride([2, 2])
        .init(device),
      conv3: UNetConvConfig::new(128, 256, 128, true).init(device),
      conv3_up: ConvTranspose2dConfig::new([128, 128], [2, 2])
        .with_stride([2, 2])
        .init(device),
      conv4: UNetConvConfig::new(128, 64, 64, true).init(device),
      conv4_up: ConvTranspose2dConfig::new([64, 64], [2, 2])
        .with_stride([2, 2])
        .init(device),
      conv5: Conv2dConfig::new([64, 64], [3, 3]).init(device),
      conv_bottom: ConvBottomConfig::new(self.out_channels, self.deconv).init(device),
      alpha: self.alpha,
    }
  }
}
