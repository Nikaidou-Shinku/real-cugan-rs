use burn::{
  module::Module,
  nn::conv::{Conv2d, ConvTranspose2d},
  tensor::{backend::Backend, ElementConversion, Tensor},
};

pub fn leaky_relu<B: Backend, const D: usize>(
  input: Tensor<B, D>,
  negative_slope: f64,
) -> Tensor<B, D> {
  let input = input.into_primitive();
  let positive_part = B::float_clamp_min(input.clone(), 0.elem());
  let negative_part = B::float_clamp_max(input, 0.elem());
  let negative_part = B::float_mul_scalar(negative_part, negative_slope.elem());
  Tensor::new(B::float_add(positive_part, negative_part))
}

#[derive(Debug, Module)]
pub enum ConvBottom<B: Backend> {
  Deconv(ConvTranspose2d<B>),
  Else(Conv2d<B>),
}

impl<B: Backend> ConvBottom<B> {
  pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
    match self {
      Self::Deconv(m) => m.forward(x),
      Self::Else(m) => m.forward(x),
    }
  }
}
