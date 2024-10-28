use burn::{
  nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
  prelude::*,
};

#[derive(Debug, Module)]
pub enum ConvBottom<B: Backend> {
  Deconv(ConvTranspose2d<B>),
  Else(Conv2d<B>),
}

impl<B: Backend> ConvBottom<B> {
  pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
    match self {
      Self::Deconv(model) => model.forward(x),
      Self::Else(model) => model.forward(x),
    }
  }
}

#[derive(Config)]
pub struct ConvBottomConfig {
  out_channels: usize,
  deconv: bool,
}

impl ConvBottomConfig {
  pub fn init<B: Backend>(&self, device: &B::Device) -> ConvBottom<B> {
    if self.deconv {
      ConvBottom::Deconv(
        ConvTranspose2dConfig::new([64, self.out_channels], [4, 4])
          .with_padding([3, 3])
          .with_stride([2, 2])
          .init(device),
      )
    } else {
      ConvBottom::Else(Conv2dConfig::new([64, self.out_channels], [3, 3]).init(device))
    }
  }
}
