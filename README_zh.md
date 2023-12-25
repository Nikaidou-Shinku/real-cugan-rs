## Real-CUGAN-rs

[English](./README.md) | 中文

---

[Real-CUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) 的 Rust 移植。

### 使用方法

将 `input.png` 超分 2 倍后保存到 `output.png`：

```shell
real-cugan-rs -i input.png -o output.png
```

完整帮助文本：

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

支持的图片格式：BMP、JPEG、PNG、WebP。

### 注意事项

- 目前仅支持 pro 模型。
- 目前 GPU 推理仅通过 CUDA 和 cuDNN 支持 NVIDIA 显卡。
- 考虑到编码速度，WebP 默认输出有损压缩图片，如果你需要无损压缩，请使用 `--lossless` 或 `-l`。
- 关于 *alpha 参数*的解释：`该值越大 AI 修复程度、痕迹越小，越模糊；alpha 越小处理越烈，越锐化，色偏（对比度、饱和度增强）越大；默认为 1.0 不调整，推荐调整区间 (0.7, 1.3)`。
- 关于 *tile size 参数*的解释：通过 `--tile-size` 或 `-t` 指定 tile size 后，图片将切分成长宽不超过 tile size 的小块进行推理。
  - 这样做会**显著减少显存占用**，一般 tile size 越小显存占用也越小，但同时**推理时间将会变长**。
  - 注意 tile size 不宜过小，一般建议不要小于 32。
  - 不指定 tile size 时，会直接使用整张图片进行推理。
- 关于 _cache_ 的解释：如果调整 tile size 后显存仍然不足，可以考虑通过 `--no-cache` 禁用对中间结果的缓存。
  - 这样做会**显著减少显存占用**，禁用缓存后只要 tile size 足够小，一般 1.5GiB 显存可以处理任意分辨率的图片。
  - 禁用缓存将**显著增加推理时间**，一般会增加到启用缓存时的 2 到 3 倍。
  - 没有指定 tile size 时，该选项将被无视。
- **欢迎 PR！**
