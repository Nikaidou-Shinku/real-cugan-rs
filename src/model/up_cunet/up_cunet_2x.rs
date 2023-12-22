use candle_core::{DType, IndexOp, Module, Tensor};
use candle_nn::VarBuilder;
use smallvec::{smallvec, SmallVec};

use crate::{
  model::unet::{UNet1, UNet2},
  utils::TensorExt,
};

pub struct UpCunet2x {
  unet1: UNet1,
  unet2: UNet2,
  alpha: f64,
  tile_size: Option<usize>,
  use_cache: bool,
}

impl UpCunet2x {
  pub fn new(
    in_channels: usize,
    out_channels: usize,
    alpha: f64,
    tile_size: Option<usize>,
    use_cache: bool,
    vb: VarBuilder,
  ) -> Result<Self, candle_core::Error> {
    let unet1 = UNet1::new(in_channels, out_channels, true, vb.pp("unet1"))?;
    let unet2 = UNet2::new(in_channels, out_channels, false, alpha, vb.pp("unet2"))?;

    if let Some(tile_size) = tile_size {
      if tile_size % 2 != 0 {
        return Err(candle_core::Error::Msg("tile_size must be divisible by 2".to_owned()).bt());
      }
    }

    Ok(Self {
      unet1,
      unet2,
      alpha,
      tile_size,
      use_cache,
    })
  }
}

impl Module for UpCunet2x {
  fn forward(&self, x: &Tensor) -> Result<Tensor, candle_core::Error> {
    if let Some(tile_size) = self.tile_size {
      return self.forward_tile(x, tile_size);
    }

    let (_, _, h0, w0) = x.shape().dims4()?;

    let ph = ((h0 - 1) / 2 + 1) * 2;
    let pw = ((w0 - 1) / 2 + 1) * 2;

    let mut x = x
      .reflection_pad(3, 18, 18 + pw - w0)?
      .reflection_pad(2, 18, 18 + ph - h0)?;

    x = self.unet1.forward(&x)?;

    let x0 = self.unet2.forward(&x)?;

    x = x
      .narrow(3, 20, x.dim(3)? - 40)?
      .narrow(2, 20, x.dim(2)? - 40)?;

    x = x0.add(&x)?;

    if w0 != pw || h0 != ph {
      x = x.narrow(3, 0, w0 * 2)?.narrow(2, 0, h0 * 2)?;
    }

    ((x - 0.15)? * (255. / 0.7))?.round()
  }
}

impl UpCunet2x {
  fn forward_tile(&self, x: &Tensor, tile_size: usize) -> Result<Tensor, candle_core::Error> {
    let (_, _, h0, w0) = x.shape().dims4()?;

    let h_tiles = (h0 - 1) / tile_size + 1;
    let w_tiles = (w0 - 1) / tile_size + 1;

    let ph = h_tiles * tile_size;
    let pw = w_tiles * tile_size;

    let x = x
      .reflection_pad(3, 18, 18 + pw - w0)?
      .reflection_pad(2, 18, 18 + ph - h0)?;

    let (n, c, h, w) = x.shape().dims4()?;

    // FIXME: we will have this Vec even if cache disabled
    let mut cache: Vec<SmallVec<[Tensor; 4]>> = Vec::with_capacity(h_tiles * w_tiles);

    let tile_num: u32 = (h_tiles * w_tiles).try_into()?;
    let tile_num: f64 = tile_num.into();

    // Stage 1
    let mut se_mean0 = Tensor::zeros((n, 64, 1, 1), DType::F32, x.device())?;

    for i in (0..(h - 36)).step_by(tile_size) {
      for j in (0..(w - 36)).step_by(tile_size) {
        let (tmp0, x_crop) = self.unet1.forward_a(&x.i((
          ..,
          ..,
          i..(i + tile_size + 36),
          j..(j + tile_size + 36),
        ))?)?;

        let tmp_se_mean = x_crop.mean_keepdim((2, 3))?;
        se_mean0 = (se_mean0 + tmp_se_mean)?;

        if self.use_cache {
          cache.push(smallvec![tmp0, x_crop]);
        }
      }
    }

    se_mean0 = (se_mean0 / tile_num)?;
    tracing::info!("Stage 1 finished");

    // Stage 2
    let mut se_mean1 = Tensor::zeros((n, 128, 1, 1), DType::F32, x.device())?;

    let Some(seblock12) = &self.unet1.conv2.seblock else {
      return Err(candle_core::Error::Msg("`unet1.conv2` has no seblock".to_owned()).bt());
    };

    for i in (0..(h - 36)).step_by(tile_size) {
      for j in (0..(w - 36)).step_by(tile_size) {
        let idx = (i / tile_size) * w_tiles + (j / tile_size);

        let (tmp0, mut x_crop) = if self.use_cache {
          let res = &cache[idx];
          (res[0].clone(), res[1].clone())
        } else {
          self.unet1.forward_a(&x.i((
            ..,
            ..,
            i..(i + tile_size + 36),
            j..(j + tile_size + 36),
          ))?)?
        };

        x_crop = seblock12.forward_mean(&x_crop, &se_mean0)?;
        let opt_unet1 = self.unet1.forward_b(&tmp0, &x_crop)?;
        let (tmp_x1, tmp_x2) = self.unet2.forward_a(&opt_unet1)?;

        let tmp_se_mean = tmp_x2.mean_keepdim((2, 3))?;
        se_mean1 = (se_mean1 + tmp_se_mean)?;

        if self.use_cache {
          cache[idx] = smallvec![opt_unet1, tmp_x1, tmp_x2];
        }
      }
    }

    se_mean1 = (se_mean1 / tile_num)?;
    tracing::info!("Stage 2 finished");

    // Stage 3
    let mut se_mean2 = Tensor::zeros((n, 128, 1, 1), DType::F32, x.device())?;

    let Some(seblock22) = &self.unet2.conv2.seblock else {
      return Err(candle_core::Error::Msg("`unet2.conv2` has no seblock".to_owned()).bt());
    };

    for i in (0..(h - 36)).step_by(tile_size) {
      for j in (0..(w - 36)).step_by(tile_size) {
        let idx = (i / tile_size) * w_tiles + (j / tile_size);

        let (opt_unet1, tmp_x1, mut tmp_x2) = if self.use_cache {
          let res = &cache[idx];
          (res[0].clone(), res[1].clone(), res[2].clone())
        } else {
          let (tmp0, mut x_crop) = self.unet1.forward_a(&x.i((
            ..,
            ..,
            i..(i + tile_size + 36),
            j..(j + tile_size + 36),
          ))?)?;

          x_crop = seblock12.forward_mean(&x_crop, &se_mean0)?;
          let opt_unet1 = self.unet1.forward_b(&tmp0, &x_crop)?;
          let (tmp_x1, tmp_x2) = self.unet2.forward_a(&opt_unet1)?;

          (opt_unet1, tmp_x1, tmp_x2)
        };

        tmp_x2 = seblock22.forward_mean(&tmp_x2, &se_mean1)?;
        let (tmp_x2, tmp_x3) = self.unet2.forward_b(&tmp_x2)?;

        let tmp_se_mean = tmp_x3.mean_keepdim((2, 3))?;
        se_mean2 = (se_mean2 + tmp_se_mean)?;

        if self.use_cache {
          cache[idx] = smallvec![opt_unet1, tmp_x1, tmp_x2, tmp_x3];
        }
      }
    }

    se_mean2 = (se_mean2 / tile_num)?;
    tracing::info!("Stage 3 finished");

    // Stage 4
    let mut se_mean3 = Tensor::zeros((n, 64, 1, 1), DType::F32, x.device())?;

    let Some(seblock23) = &self.unet2.conv3.seblock else {
      return Err(candle_core::Error::Msg("`unet2.conv3` has no seblock".to_owned()).bt());
    };

    for i in (0..(h - 36)).step_by(tile_size) {
      for j in (0..(w - 36)).step_by(tile_size) {
        let idx = (i / tile_size) * w_tiles + (j / tile_size);

        let (opt_unet1, tmp_x1, tmp_x2, mut tmp_x3) = if self.use_cache {
          let res = &cache[idx];
          (
            res[0].clone(),
            res[1].clone(),
            res[2].clone(),
            res[3].clone(),
          )
        } else {
          let (tmp0, mut x_crop) = self.unet1.forward_a(&x.i((
            ..,
            ..,
            i..(i + tile_size + 36),
            j..(j + tile_size + 36),
          ))?)?;

          x_crop = seblock12.forward_mean(&x_crop, &se_mean0)?;
          let opt_unet1 = self.unet1.forward_b(&tmp0, &x_crop)?;
          let (tmp_x1, mut tmp_x2) = self.unet2.forward_a(&opt_unet1)?;
          tmp_x2 = seblock22.forward_mean(&tmp_x2, &se_mean1)?;
          let (tmp_x2, tmp_x3) = self.unet2.forward_b(&tmp_x2)?;

          (opt_unet1, tmp_x1, tmp_x2, tmp_x3)
        };

        tmp_x3 = seblock23.forward_mean(&tmp_x3, &se_mean2)?;
        let mut tmp_x4 = self.unet2.forward_c(&tmp_x2, &tmp_x3)?;
        tmp_x4 = (tmp_x4 * self.alpha)?;

        let tmp_se_mean = tmp_x4.mean_keepdim((2, 3))?;
        se_mean3 = (se_mean3 + tmp_se_mean)?;

        if self.use_cache {
          cache[idx] = smallvec![opt_unet1, tmp_x1, tmp_x4];
        }
      }
    }

    se_mean3 = (se_mean3 / tile_num)?;
    tracing::info!("Stage 4 finished");

    // Stage tail
    let mut res = Tensor::zeros((n, c, h * 2 - 72, w * 2 - 72), DType::F32, x.device())?;

    let Some(seblock24) = &self.unet2.conv4.seblock else {
      return Err(candle_core::Error::Msg("`unet2.conv4` has no seblock".to_owned()).bt());
    };

    for i in (0..(h - 36)).step_by(tile_size) {
      for j in (0..(w - 36)).step_by(tile_size) {
        let idx = (i / tile_size) * w_tiles + (j / tile_size);

        let (mut x_crop, tmp_x1, mut tmp_x4) = if self.use_cache {
          let res = &cache[idx];
          (res[0].clone(), res[1].clone(), res[2].clone())
        } else {
          let (tmp0, mut x_crop) = self.unet1.forward_a(&x.i((
            ..,
            ..,
            i..(i + tile_size + 36),
            j..(j + tile_size + 36),
          ))?)?;

          x_crop = seblock12.forward_mean(&x_crop, &se_mean0)?;
          x_crop = self.unet1.forward_b(&tmp0, &x_crop)?;
          let (tmp_x1, mut tmp_x2) = self.unet2.forward_a(&x_crop)?;
          tmp_x2 = seblock22.forward_mean(&tmp_x2, &se_mean1)?;
          let (tmp_x2, mut tmp_x3) = self.unet2.forward_b(&tmp_x2)?;
          tmp_x3 = seblock23.forward_mean(&tmp_x3, &se_mean2)?;
          let mut tmp_x4 = self.unet2.forward_c(&tmp_x2, &tmp_x3)?;
          // TODO: check this
          tmp_x4 = (tmp_x4 * self.alpha)?;

          (x_crop, tmp_x1, tmp_x4)
        };

        x_crop = x_crop
          .narrow(3, 20, x_crop.dim(3)? - 40)?
          .narrow(2, 20, x_crop.dim(2)? - 40)?;

        tmp_x4 = seblock24.forward_mean(&tmp_x4, &se_mean3)?;
        let x0 = self.unet2.forward_d(&tmp_x1, &tmp_x4)?;
        x_crop = x0.add(&x_crop)?;

        res = res.slice_assign(
          &[
            0..n,
            0..c,
            (i * 2)..(i * 2 + tile_size * 2),
            (j * 2)..(j * 2 + tile_size * 2),
          ],
          &((x_crop - 0.15)? * (255. / 0.7))?.round()?,
        )?;
      }
    }

    if w0 != pw || h0 != ph {
      res = res.narrow(3, 0, w0 * 2)?.narrow(2, 0, h0 * 2)?;
    }

    Ok(res)
  }
}
