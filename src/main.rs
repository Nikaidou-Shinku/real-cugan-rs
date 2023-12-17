mod model;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use image::{io::Reader as ImageReader, ImageBuffer};

use model::UpCunet2x;

fn main() -> Result<(), candle_core::Error> {
  let device = Device::new_cuda(0)?;

  // TODO: args
  let img = ImageReader::open("input.png")
    .expect("Failed to open image file")
    .decode()
    .expect("Failed to decode image file");

  let raw = img
    .to_rgb8()
    .into_raw()
    .into_iter()
    .map(|x| x as f32)
    .collect();

  let mut data = Tensor::from_vec(
    raw,
    (img.height() as usize, img.width() as usize, 3),
    &device,
  )?
  .permute((2, 0, 1))?
  .unsqueeze(0)?;
  data = ((data / (255. / 0.7))? + 0.15)?;

  let vb = VarBuilder::from_pth("./models/pro-no-denoise-up2x.pth", DType::F32, &device)?;
  let model = UpCunet2x::new(vb, 3, 3)?;

  let res = model.forward(&data)?;
  let res = res.squeeze(0)?.permute((1, 2, 0))?;

  let mut imgbuf = ImageBuffer::new(res.dim(1)? as u32, res.dim(0)? as u32);

  let res: Vec<Vec<Vec<f32>>> = res.to_vec3()?;

  for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
    *pixel = image::Rgb(
      res[y as usize][x as usize]
        .iter()
        .map(|&x| x as u8)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap(),
    );
  }

  imgbuf
    .save("output.png")
    .expect("Failed to write image file");

  Ok(())
}
