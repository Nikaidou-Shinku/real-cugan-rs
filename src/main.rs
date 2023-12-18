mod cli;
mod model;
mod setup;

use std::env;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use image::{io::Reader as ImageReader, ImageBuffer};

use cli::Cli;
use model::UpCunet2x;
use setup::setup_tracing;

fn main() -> Result<(), candle_core::Error> {
  setup_tracing();

  let args = Cli::parse();

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

  let raw: Vec<f32> = img
    .to_rgb8()
    .into_raw()
    .into_iter()
    .map(Into::into)
    .collect();

  let mut data = Tensor::from_vec(raw, (height, width, 3), &device)?
    .permute((2, 0, 1))?
    .unsqueeze(0)?;
  data = ((data / (255. / 0.7))? + 0.15)?;

  tracing::info!("Preprocess the image into tensor");

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
  let model = UpCunet2x::new(3, 3, vb)?;

  tracing::info!("Network built");

  let res = model.forward(&data)?;
  let res = res.squeeze(0)?.permute((1, 2, 0))?;

  tracing::info!("Image processed");

  let mut imgbuf = ImageBuffer::new(res.dim(1)?.try_into()?, res.dim(0)?.try_into()?);

  let res: Vec<Vec<Vec<f32>>> = res.to_vec3()?;

  for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
    let x: usize = x.try_into()?;
    let y: usize = y.try_into()?;

    let p: Vec<_> = res[y][x].iter().map(|v| v.clamp(0., 255.) as u8).collect();

    *pixel = image::Rgb(p.try_into().unwrap());
  }

  tracing::info!("Convert the tensor to image");

  imgbuf
    .save(&args.output_path)
    .expect("Failed to write image file");

  tracing::info!(path = ?args.output_path, "Image saved");

  Ok(())
}
