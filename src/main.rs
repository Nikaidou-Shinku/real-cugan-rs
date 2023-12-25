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
use model::{RealCugan, UpCunet2x, UpCunet3x};
use setup::{setup_args, setup_tracing};
use utils::{preprocess_alpha_channel, save_image};

fn main() -> Result<(), candle_core::Error> {
  setup_tracing();

  let args = Cli::parse();

  let output_format = match setup_args(&args) {
    Ok(res) => res,
    Err(err) => {
      tracing::error!("{err}");
      return Ok(());
    }
  };

  let model_name = format!(
    "pro-{}-up{}x.pth",
    match args.denoise_level.as_str() {
      "-1" => "conservative".to_owned(),
      "0" => "no-denoise".to_owned(),
      d => format!("denoise{d}x"),
    },
    args.scale
  );

  let model_path = env::current_exe()?
    .parent()
    .expect("Failed to get parent directory of the executable")
    .join("models")
    .join(&model_name);

  if !model_path.is_file() {
    tracing::error!(model_name, "Failed to find the model");
    return Ok(());
  }

  let img = ImageReader::open(args.input_path)
    .expect("Failed to open image file")
    .decode()
    .expect("Failed to decode image file");

  let width: usize = img.width().try_into()?;
  let height: usize = img.height().try_into()?;

  tracing::info!(width, height, "Image file read");

  let device = if args.use_cpu {
    Device::Cpu
  } else {
    Device::new_cuda(0)?
  };

  tracing::info!(?device, "Setup device");

  // TODO: Reduce redundancy here
  let (rgb, alpha) = match img {
    DynamicImage::ImageRgb8(img) => {
      tracing::info!("No alpha channel found");

      (
        Tensor::from_vec(img.into_raw(), (height, width, 3), &device)?.to_dtype(DType::F32)?,
        None,
      )
    }
    DynamicImage::ImageRgba8(img) => {
      if output_format == ImageFormat::Jpeg {
        tracing::error!("Images in JPEG format cannot save transparent layers!");
        return Ok(());
      }

      tracing::info!("Preprocess the alpha channel...");

      let data = Tensor::from_vec(img.into_raw(), (height, width, 4), &device)?;
      let (rgb, alpha) = preprocess_alpha_channel(&data)?;
      (rgb, Some(alpha))
    }
    others => {
      if output_format == ImageFormat::Jpeg {
        tracing::warn!("The output format is JPEG, Convert into RGB...");
        let img = others.to_rgb8();

        (
          Tensor::from_vec(img.into_raw(), (height, width, 3), &device)?.to_dtype(DType::F32)?,
          None,
        )
      } else {
        tracing::warn!("Convert into RGBA...");
        let img = others.to_rgba8();

        tracing::info!("Preprocess the alpha channel...");

        let data = Tensor::from_vec(img.into_raw(), (height, width, 4), &device)?;
        let (rgb, alpha) = preprocess_alpha_channel(&data)?;
        (rgb, Some(alpha))
      }
    }
  };

  let data = rgb.permute((2, 0, 1))?.unsqueeze(0)?;
  let data = ((data / (255. / 0.7))? + 0.15)?; // for pro model

  tracing::info!(
    has_alpha = alpha.is_some(),
    "Preprocess the image into tensor",
  );

  let (target_width, target_height) = {
    let scale: usize = args.scale.into();
    (width * scale, height * scale)
  };

  let alpha = alpha.map(|alpha| {
    let mut resizer = resize::new(
      width,
      height,
      target_width,
      target_height,
      Pixel::Gray8,
      resize::Type::Mitchell,
    )
    .expect("Failed to initialize the alpha channel resizer");

    let mut dst = vec![0; target_width * target_height];

    resizer
      .resize(alpha.as_gray(), dst.as_gray_mut())
      .expect("Failed to upscale the alpha channel");

    tracing::info!("Alpha channel processed");

    dst
  });

  let vb = VarBuilder::from_pth(model_path, DType::F32, &device)?;
  let model = match args.scale {
    2 => RealCugan::X2(UpCunet2x::new(
      3,
      3,
      args.alpha,
      args.tile_size,
      !args.no_cache,
      vb,
    )?),
    3 => RealCugan::X3(UpCunet3x::new(3, 3, args.alpha, args.tile_size, vb)?),
    _ => {
      tracing::error!(scale = args.scale, "Unsupported upscale ratio");
      return Ok(());
    }
  };

  tracing::info!("Network built");

  let res = model.forward(&data)?;
  drop(data);

  tracing::info!("Real-CUGAN finished");

  let res = ((res - 0.15)? * (255. / 0.7))?.round()?; // for pro model
  let res = res.squeeze(0)?.permute((1, 2, 0))?;

  save_image(
    target_width,
    target_height,
    &res,
    alpha,
    &args.output_path,
    output_format,
    args.lossless,
  )?;

  tracing::info!(path = ?args.output_path, "Image saved");

  Ok(())
}
