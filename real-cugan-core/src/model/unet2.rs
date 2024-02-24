use burn::{
  config::Config,
  module::Module,
  nn::conv::{Conv2d, ConvTranspose2d},
  tensor::{backend::Backend, Tensor},
};

use super::{unet_conv::UNetConv, utils::ConvBottom};

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
    todo!()
  }
}

#[derive(Config)]
pub struct UNet2Config {
  in_channels: usize,
  out_channels: usize,
  alpha: f64,
}

impl UNet2Config {
  pub fn init_with<B: Backend>(&self, record: UNet2Record<B>) -> UNet2<B> {
    todo!()
  }
}
