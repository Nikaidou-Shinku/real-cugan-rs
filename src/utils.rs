use std::{fs::File, io::BufWriter, path::Path};

use candle_core::{shape::Dim, DType, Tensor};
use image::{
  codecs::{
    bmp::BmpEncoder,
    jpeg::JpegEncoder,
    png::{self, PngEncoder},
    webp::{self, WebPEncoder},
  },
  ColorType, ImageEncoder, ImageFormat,
};

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
        v.push(self.narrow(dim, self.dim(dim)? - i - 2, 1)?)
      }
      Tensor::cat(&v, dim)
    } else if right == 0 {
      let dim = dim.to_index(self.shape(), "reflection_pad")?;
      let mut v = vec![];
      for i in 0..left {
        v.push(self.narrow(dim, left - i, 1)?)
      }
      v.push(self.clone());
      Tensor::cat(&v, dim)
    } else {
      let dim = dim.to_index(self.shape(), "reflection_pad")?;
      let mut v = vec![];
      for i in 0..left {
        v.push(self.narrow(dim, left - i, 1)?)
      }
      v.push(self.clone());
      for i in 0..right {
        v.push(self.narrow(dim, self.dim(dim)? - i - 2, 1)?)
      }
      Tensor::cat(&v, dim)
    }
  }
}

pub fn preprocess_alpha_channel(data: &Tensor) -> Result<(Tensor, Vec<u8>), candle_core::Error> {
  let rgb = data.narrow(2, 0, 3)?.to_dtype(DType::F32)?;
  let alpha = data.narrow(2, 3, 1)?;

  let raw_alpha: Vec<u8> = alpha.flatten_all()?.to_vec1()?;
  let alpha_mask = (Tensor::cat(&[&alpha, &alpha, &alpha], 2)?.to_dtype(DType::F32)? / 255.)?;

  return Ok(((rgb * alpha_mask)?, raw_alpha));
}

pub fn save_image(
  width: usize,
  height: usize,
  rgb: &Tensor,
  alpha: Option<Vec<u8>>,
  path: impl AsRef<Path>,
  format: ImageFormat,
  lossless: bool,
) -> Result<(), candle_core::Error> {
  let (buffer, color_type) = if let Some(alpha) = alpha {
    let alpha = Tensor::from_vec(alpha, (height, width, 1), rgb.device())?;
    let alpha_mask = (255. / Tensor::cat(&[&alpha, &alpha, &alpha], 2)?.to_dtype(DType::F32)?)?;

    let rgb = (rgb * alpha_mask)?.clamp(0., 255.)?.to_dtype(DType::U8)?;

    (
      Tensor::cat(&[rgb, alpha], 2)?.flatten_all()?.to_vec1()?,
      ColorType::Rgba8,
    )
  } else {
    (
      rgb
        .clamp(0., 255.)?
        .to_dtype(DType::U8)?
        .flatten_all()?
        .to_vec1()?,
      ColorType::Rgb8,
    )
  };

  let width = width.try_into()?;
  let height = height.try_into()?;

  let mut buffered_file_write =
    BufWriter::new(File::create(path).expect("Failed to create output image file"));

  match format {
    ImageFormat::Bmp => {
      if !lossless {
        tracing::warn!("BMP images cannot be lossy, output lossless result...");
      }

      BmpEncoder::new(&mut buffered_file_write).write_image(&buffer, width, height, color_type)
    }

    ImageFormat::Jpeg => {
      if lossless {
        panic!("JPEG images cannot be lossless");
      }

      if color_type == ColorType::Rgba8 {
        panic!("Images in JPEG format cannot save transparent layers!");
      }

      JpegEncoder::new_with_quality(buffered_file_write, 100)
        .write_image(&buffer, width, height, color_type)
    }

    ImageFormat::Png => {
      if !lossless {
        tracing::warn!("PNG images cannot be lossy, output lossless result...");
      }

      PngEncoder::new_with_quality(
        buffered_file_write,
        png::CompressionType::Fast,
        png::FilterType::Adaptive,
      )
      .write_image(&buffer, width, height, color_type)
    }

    ImageFormat::WebP => WebPEncoder::new_with_quality(
      buffered_file_write,
      if lossless {
        webp::WebPQuality::lossless()
      } else {
        webp::WebPQuality::lossy(100)
      },
    )
    .write_image(&buffer, width, height, color_type),

    _ => {
      panic!("Unsupported output image format");
    }
  }
  .expect("Failed to encode image & write to file");

  Ok(())
}
