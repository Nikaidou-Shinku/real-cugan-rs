use crate::model::unet::{UNet1, UNet2};

struct UpCunet2x {
  unet1: UNet1,
  unet2: UNet2,
}
