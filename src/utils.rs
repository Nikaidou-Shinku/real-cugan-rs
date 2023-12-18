use candle_core::{shape::Dim, Tensor};

pub trait TensorExt {
  fn reflection_pad<D: Dim>(
    &self,
    dim: D,
    left: usize,
    right: usize,
  ) -> Result<Self, candle_core::Error>
  where
    Self: Sized;
}

impl TensorExt for Tensor {
  fn reflection_pad<D: Dim>(
    &self,
    dim: D,
    left: usize,
    right: usize,
  ) -> Result<Self, candle_core::Error> {
    if left == 0 && right == 0 {
      Ok(self.clone())
    } else if self.elem_count() == 0 {
      Err(candle_core::Error::Msg("cannot use reflection_pad on an empty tensor".to_owned()).bt())
    } else if left == 0 {
      let dim = dim.to_index(self.shape(), "reflection_pad")?;
      let mut v = vec![self.clone()];
      for i in 0..right {
        v.push(self.narrow(dim, self.dim(dim)? - i - 1, 1)?)
      }
      Tensor::cat(&v, dim)
    } else if right == 0 {
      let dim = dim.to_index(self.shape(), "reflection_pad")?;
      let mut v = vec![];
      for i in 0..left {
        v.push(self.narrow(dim, left - i - 1, 1)?)
      }
      v.push(self.clone());
      Tensor::cat(&v, dim)
    } else {
      let dim = dim.to_index(self.shape(), "reflection_pad")?;
      let mut v = vec![];
      for i in 0..left {
        v.push(self.narrow(dim, left - i - 1, 1)?)
      }
      v.push(self.clone());
      for i in 0..right {
        v.push(self.narrow(dim, self.dim(dim)? - i - 1, 1)?)
      }
      Tensor::cat(&v, dim)
    }
  }
}
