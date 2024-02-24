use burn::{
  config::Config,
  module::Module,
  nn::conv::{Conv2d, Conv2dConfig},
  tensor::{backend::Backend, Tensor},
};

use super::{
  se_block::{SeBlock, SeBlockConfig},
  utils::leaky_relu,
};

#[derive(Debug, Module)]
pub struct UNetConv<B: Backend> {
  conv0: Conv2d<B>,
  conv2: Conv2d<B>,
  seblock: Option<SeBlock<B>>,
}

impl<B: Backend> UNetConv<B> {
  pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
    let x = self.conv0.forward(x);
    let x = leaky_relu(x, 0.1);
    let x = self.conv2.forward(x);
    let x = leaky_relu(x, 0.1);

    if let Some(seblock) = &self.seblock {
      seblock.forward(x)
    } else {
      x
    }
  }
}

#[derive(Config)]
pub struct UNetConvConfig {
  in_channels: usize,
  mid_channels: usize,
  out_channels: usize,
}

impl UNetConvConfig {
  pub fn init_with<B: Backend>(&self, record: UNetConvRecord<B>) -> UNetConv<B> {
    UNetConv {
      conv0: Conv2dConfig::new([self.in_channels, self.mid_channels], [3, 3])
        .init_with(record.conv0),
      conv2: Conv2dConfig::new([self.mid_channels, self.out_channels], [3, 3])
        .init_with(record.conv2),
      seblock: record.seblock.map(|record| {
        SeBlockConfig::new(self.out_channels)
          .with_bias(true)
          .init_with(record)
      }),
    }
  }
}
