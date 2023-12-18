use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
#[command(version, author)]
#[command(about = "A Rust port of Real-CUGAN", long_about = None)]
pub struct Cli {
  #[arg(short, long, help = "Input image path")]
  #[arg(value_name = "INPUT")]
  pub input_path: PathBuf,

  #[arg(short, long, help = "Output image path")]
  #[arg(value_name = "OUTPUT")]
  pub output_path: PathBuf,

  #[arg(short = 'c', long, help = "Use CPU instead of GPU for inference")]
  pub use_cpu: bool,

  #[arg(short, long, help = "Filename of the model in `models` directory")]
  #[arg(value_name = "MODEL", default_value = "pro-no-denoise-up2x.pth")]
  pub model_name: PathBuf,
}
