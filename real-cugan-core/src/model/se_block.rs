use burn::{
  config::Config,
  module::Module,
  nn::conv::{Conv2d, Conv2dConfig},
  tensor::{
    activation::{relu, sigmoid},
    backend::Backend,
    Tensor,
  },
};

#[derive(Debug, Module)]
pub struct SeBlock<B: Backend> {
  conv1: Conv2d<B>,
  conv2: Conv2d<B>,
}

impl<B: Backend> SeBlock<B> {
  pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
    let x0 = x.clone().mean_dim(2).mean_dim(3);
    let x0 = self.conv1.forward(x0);
    let x0 = relu(x0);
    let x0 = self.conv2.forward(x0);
    let x0 = sigmoid(x0);
    x.mul(x0)
  }
}

#[derive(Config)]
pub struct SeBlockConfig {
  in_channels: usize,
  #[config(default = 8)]
  reduction: usize,
  #[config(default = false)]
  bias: bool,
}

impl SeBlockConfig {
  pub fn init_with<B: Backend>(&self, record: SeBlockRecord<B>) -> SeBlock<B> {
    SeBlock {
      conv1: Conv2dConfig::new(
        [self.in_channels, self.in_channels / self.reduction],
        [1, 1],
      )
      .with_bias(self.bias)
      .init_with(record.conv1),
      conv2: Conv2dConfig::new(
        [self.in_channels / self.reduction, self.in_channels],
        [1, 1],
      )
      .with_bias(self.bias)
      .init_with(record.conv2),
    }
  }
}
