use std::{fs::File, io::BufWriter, path::Path};

use candle_core::{shape::Dim, Tensor};
use image::{
  codecs::{
    bmp::BmpEncoder,
    jpeg::JpegEncoder,
    png::{self, PngEncoder},
    webp::{self, WebPEncoder},
  },
  ColorType, ImageEncoder, ImageFormat, RgbaImage,
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

pub fn preprocess_alpha_channel(
  image: RgbaImage,
) -> Result<(Vec<f32>, Option<Vec<u8>>), &'static str> {
  let raw = image.into_raw();

  if raw.len() % 4 != 0 {
    return Err("Failed to preprocess the alpha channel! `len % 4 != 0`");
  }

  let (rgb, alpha): (Vec<[u8; 3]>, Vec<u8>) = raw
    .chunks_exact(4)
    .map(|rgba| ([rgba[0], rgba[1], rgba[2]], rgba[3]))
    .unzip();

  let rgb = rgb.into_iter().flatten().map(Into::into).collect();

  Ok((rgb, Some(alpha)))
}

pub fn save_image(
  width: u32,
  height: u32,
  rgb: Vec<f32>,
  alpha: Option<Vec<u8>>,
  path: impl AsRef<Path>,
  format: ImageFormat,
  lossless: bool,
) -> Result<(), &'static str> {
  let (buffer, color_type): (Vec<_>, _) = if let Some(alpha) = alpha {
    (
      rgb
        .into_iter()
        .map(|v| v.clamp(0., 255.) as u8)
        .array_chunks::<3>()
        .zip(alpha)
        .map(|(x, y)| [x[0], x[1], x[2], y])
        .flatten()
        .collect(),
      ColorType::Rgba8,
    )
  } else {
    (
      rgb.into_iter().map(|v| v.clamp(0., 255.) as u8).collect(),
      ColorType::Rgb8,
    )
  };

  tracing::info!("Convert the tensor to image");

  let mut buffered_file_write =
    BufWriter::new(File::create(path).map_err(|_| "Failed to create output image file")?);

  match format {
    ImageFormat::Bmp => {
      if !lossless {
        tracing::warn!("BMP images cannot be lossy, output lossless result...");
      }

      BmpEncoder::new(&mut buffered_file_write).write_image(&buffer, width, height, color_type)
    }

    ImageFormat::Jpeg => {
      if lossless {
        return Err("JPEG images cannot be lossless");
      }

      if color_type == ColorType::Rgba8 {
        return Err("Images in JPEG format cannot save transparent layers!");
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
      return Err("Unsupported output image format");
    }
  }
  .map_err(|_| "Failed to encode image & write to file")?;

  Ok(())
}
