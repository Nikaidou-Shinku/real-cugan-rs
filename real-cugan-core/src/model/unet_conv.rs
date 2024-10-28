use burn::{
  nn::conv::{Conv2d, Conv2dConfig},
  prelude::*,
  tensor::activation::leaky_relu,
};

use super::{SeBlock, SeBlockConfig};

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
  se: bool,
}

impl UNetConvConfig {
  pub fn init<B: Backend>(&self, device: &B::Device) -> UNetConv<B> {
    UNetConv {
      conv0: Conv2dConfig::new([self.in_channels, self.mid_channels], [3, 3]).init(device),
      conv2: Conv2dConfig::new([self.mid_channels, self.out_channels], [3, 3]).init(device),
      seblock: if self.se {
        Some(
          SeBlockConfig::new(self.out_channels)
            .with_bias(true)
            .init(device),
        )
      } else {
        None
      },
    }
  }
}
