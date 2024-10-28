use burn::prelude::*;

use super::{UNet1, UNet1Config, UNet2, UNet2Config};

#[derive(Debug, Module)]
pub struct UpCunet2x<B: Backend> {
  unet1: UNet1<B>,
  unet2: UNet2<B>,
  alpha: f64,
}

impl<B: Backend> UpCunet2x<B> {
  pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [_, _, h0, w0] = x.dims();

    let ph = ((h0 - 1) / 2 + 1) * 2;
    let pw = ((w0 - 1) / 2 + 1) * 2;

    // TODO: reflection padding
    let x = x.pad((18, 18 + pw - w0, 18, 18 + ph - h0), 0.elem());

    let x = self.unet1.forward(x);
    let x0 = self.unet2.forward(x.clone());

    let x = {
      let [_, _, h, w] = x.dims();
      x.narrow(3, 20, w - 40).narrow(2, 20, h - 40)
    };

    let x = x0 + x;

    if w0 != pw || h0 != ph {
      x.narrow(3, 0, w0 * 2).narrow(2, 0, h0 * 2)
    } else {
      x
    }
  }
}

#[derive(Config)]
pub struct UpCunet2xConfig {
  #[config(default = 3)]
  in_channels: usize,
  #[config(default = 3)]
  out_channels: usize,
  #[config(default = 1.0)]
  alpha: f64,
}

impl UpCunet2xConfig {
  pub fn init<B: Backend>(&self, device: &B::Device) -> UpCunet2x<B> {
    UpCunet2x {
      unet1: UNet1Config::new(self.in_channels, self.out_channels, true).init(device),
      unet2: UNet2Config::new(self.in_channels, self.out_channels, false).init(device),
      alpha: self.alpha,
    }
  }
}
