mod cli;
mod model;
mod setup;
mod utils;

use std::env;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use image::{io::Reader as ImageReader, DynamicImage, ImageFormat};
use resize::Pixel;
use rgb::FromSlice;

use cli::Cli;
use model::UpCunet2x;
use setup::{setup_args, setup_tracing};
use utils::{preprocess_alpha_channel, save_image};

fn main() -> Result<(), candle_core::Error> {
  setup_tracing();

  let args = Cli::parse();

  let image_format = match setup_args(&args) {
    Ok(res) => res,
    Err(err) => {
      tracing::error!("{err}");
      return Ok(());
    }
  };

  let device = if args.use_cpu {
    Device::Cpu
  } else {
    Device::new_cuda(0)?
  };

  tracing::info!(?device, "Setup device");

  let img = ImageReader::open(args.input_path)
    .expect("Failed to open image file")
    .decode()
    .expect("Failed to decode image file");

  let width: usize = img.width().try_into()?;
  let height: usize = img.height().try_into()?;

  tracing::info!(width, height, "Image file read");

  let (rgb, alpha) = match img {
    DynamicImage::ImageRgb8(img) => {
      tracing::info!("No alpha channel found");
      (img.into_raw().into_iter().map(Into::into).collect(), None)
    }
    DynamicImage::ImageRgba8(img) => {
      tracing::info!("Preprocess the alpha channel...");

      match preprocess_alpha_channel(img) {
        Ok(res) => res,
        Err(msg) => {
          tracing::error!("{msg}");
          return Ok(());
        }
      }
    }
    others => {
      tracing::warn!("Convert into RGBA...");
      let img = others.to_rgba8();

      tracing::info!("Preprocess the alpha channel...");
      match preprocess_alpha_channel(img) {
        Ok(res) => res,
        Err(msg) => {
          tracing::error!("{msg}");
          return Ok(());
        }
      }
    }
  };

  if alpha.is_some() {
    if image_format == ImageFormat::Jpeg {
      tracing::error!("Images in JPEG format cannot save transparent layers!");
      return Ok(());
    }

    tracing::info!("Alpha channel preprocessed");
  }

  let mut data = Tensor::from_vec(rgb, (height, width, 3), &device)?
    .permute((2, 0, 1))?
    .unsqueeze(0)?;
  data = ((data / (255. / 0.7))? + 0.15)?;

  tracing::info!("Preprocess the rgb channel into tensor");

  let model_path = env::current_exe()?
    .parent()
    .expect("Failed to get parent directory of the executable")
    .join("models")
    .join(args.model_name);

  if !model_path.is_file() {
    tracing::error!(?model_path, "Failed to find the model");
    return Ok(());
  }

  let vb = VarBuilder::from_pth(model_path, DType::F32, &device)?;
  let model = UpCunet2x::new(3, 3, args.alpha, args.tile_size, !args.no_cache, vb)?;

  tracing::info!("Network built");

  let res = model.forward(&data)?;
  let res = res.squeeze(0)?.permute((1, 2, 0))?;

  tracing::info!("Rgb channel processed");

  let alpha = alpha.map(|alpha| {
    let mut resizer = resize::new(
      width,
      height,
      width * 2,
      height * 2,
      Pixel::Gray8,
      resize::Type::Mitchell,
    )
    .expect("Failed to initialize the alpha channel resizer");

    let mut dst = vec![0; width * height * 4];

    resizer
      .resize(alpha.as_gray(), dst.as_gray_mut())
      .expect("Failed to upscale the alpha channel");

    tracing::info!("Alpha channel processed");

    dst
  });

  let res: Vec<f32> = res.flatten_all()?.to_vec1()?;

  if let Err(msg) = save_image(
    (width * 2).try_into()?,
    (height * 2).try_into()?,
    res,
    alpha,
    &args.output_path,
    image_format,
    args.lossless,
  ) {
    tracing::error!(msg, "Failed to save image");
  } else {
    tracing::info!(path = ?args.output_path, "Image saved");
  }

  Ok(())
}
