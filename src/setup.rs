use image::ImageFormat;

use tracing::Level;
use tracing_subscriber::FmtSubscriber;

use crate::cli::Cli;

pub fn setup_tracing() {
  let subscriber = FmtSubscriber::builder()
    .with_max_level(Level::INFO)
    .with_target(false)
    .finish();

  tracing::subscriber::set_global_default(subscriber).expect("Setting default subscriber failed");
}

pub fn setup_args(args: &Cli) -> Result<ImageFormat, &'static str> {
  if args.no_cache && args.tile_size.is_none() {
    tracing::warn!("Cache only works with tile mode! Ignoring `--no-cache`...");
  }

  let Ok(output_format) = ImageFormat::from_path(&args.output_path) else {
    return Err("Failed to get image format from the output path");
  };

  if output_format == ImageFormat::Jpeg && args.lossless {
    return Err("JPEG images cannot be lossless");
  }

  Ok(output_format)
}
