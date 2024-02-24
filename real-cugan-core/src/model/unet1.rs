use burn::{
  config::Config,
  module::Module,
  nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
  tensor::{backend::Backend, Tensor},
};

use super::{
  unet_conv::{UNetConv, UNetConvConfig},
  utils::{leaky_relu, ConvBottom, ConvBottomRecord},
};

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

    let [_, _, h, w] = x1.dims();
    let x1 = x1.narrow(3, 4, w - 8).narrow(2, 4, h - 8);

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
  for_x3: bool,
}

impl UNet1Config {
  pub fn init_with<B: Backend>(&self, record: UNet1Record<B>) -> UNet1<B> {
    UNet1 {
      conv1: UNetConvConfig::new(self.in_channels, 32, 64).init_with(record.conv1),
      conv1_down: Conv2dConfig::new([64, 64], [2, 2])
        .with_stride([2, 2])
        .init_with(record.conv1_down),
      conv2: UNetConvConfig::new(64, 128, 64).init_with(record.conv2),
      conv2_up: ConvTranspose2dConfig::new([64, 64], [2, 2])
        .with_stride([2, 2])
        .init_with(record.conv2_up),
      conv3: Conv2dConfig::new([64, 64], [3, 3]).init_with(record.conv3),
      conv_bottom: match record.conv_bottom {
        ConvBottomRecord::Deconv(record) => {
          let kernel_size = if self.for_x3 { [5, 5] } else { [4, 4] };
          let stride = if self.for_x3 { [3, 3] } else { [2, 2] };
          let padding = if self.for_x3 { [2, 2] } else { [3, 3] };

          ConvBottom::Deconv(
            ConvTranspose2dConfig::new([64, self.out_channels], kernel_size)
              .with_stride(stride)
              .with_padding(padding)
              .init_with(record),
          )
        }
        ConvBottomRecord::Else(record) => {
          ConvBottom::Else(Conv2dConfig::new([64, self.out_channels], [3, 3]).init_with(record))
        }
      },
    }
  }
}
