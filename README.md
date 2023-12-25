## Real-CUGAN-rs

English | [中文](./README_zh.md)

---

A Rust port of [Real-CUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN).

### Usages

Upscale `input.png` to 2x and save it to `output.png`:

```shell
real-cugan-rs -i input.png -o output.png
```

Full help text:

```console
A Rust port of Real-CUGAN

Usage: real-cugan-rs [OPTIONS] --input-path <INPUT> --output-path <OUTPUT>

Options:
  -i, --input-path <INPUT>       Input image path
  -o, --output-path <OUTPUT>     Output image path
  -s, --scale <SCALE>            Upscale ratio (2/3) [default: 2]
  -d, --denoise-level <DENOISE>  Denoise level (-1/0/3), -1 for conservative model [default: 0]
  -l, --lossless                 Output lossless encoded image
  -t, --tile-size <TILE>         Tile size, smaller value may reduce memory usage
      --no-cache                 Disable cache, which increases runtime but reduce memory usage
  -C, --use-cpu                  Use CPU instead of GPU for inference
  -a, --alpha <ALPHA>            Please check the documentation for this option [default: 1.0]
  -h, --help                     Print help
  -V, --version                  Print version
```

Supported image formats: BMP, JPEG, PNG, WebP.

### Note

- Currently only the pro model is supported.
- Currently GPU inference only supports NVIDIA graphics cards through CUDA and cuDNN.
- Considering the encoding speed, WebP outputs lossy compressed images by default. If you need lossless compression, please add `--lossless` or `-l`.
- Explanation of _the alpha option_: `该值越大 AI 修复程度、痕迹越小，越模糊；alpha 越小处理越烈，越锐化，色偏（对比度、饱和度增强）越大；默认为 1.0 不调整，推荐调整区间 (0.7, 1.3)`.
- Explanation of _the tile size option_: After specifying tile size through `--tile-size` or `-t`, the image will be divided into small blocks with a length not exceeding the tile size for inference.
  - This will **significantly reduce the memory usage**. Generally, the smaller the tile size, the smaller the memory usage will be, but at the same time **the inference time will become longer**.
  - Note that the tile size should not be too small, and it is generally recommended not to be less than 32.
  - When tile size is not specified, the entire image will be used directly for inference.
- Explanation on _cache_: If the memory is still insufficient after adjusting the tile size, you can consider disabling the cache through `--no-cache`.
  - This will **significantly reduce the memory usage**. After disabling caching, as long as the tile size is small enough, generally 1.5GiB of video memory can handle images of any resolution.
  - Disabling caching will **significantly increase inference time**, typically to 2 to 3 times that with caching enabled.
  - This option is ignored when tile size is not specified.
- **PRs are welcome!**
