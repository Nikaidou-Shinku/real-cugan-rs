## Real-CUGAN-rs

[English](./README.md) | 中文

---

[Real-CUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) 的 Rust 移植。

### 使用方法

将 `input.webp` 超分后保存到 `output.webp`：

```shell
real-cugan-rs -i input.webp -o output.webp
```

完整帮助文本：

```console
A Rust port of Real-CUGAN

Usage: real-cugan-rs [OPTIONS] --input-path <INPUT> --output-path <OUTPUT>

Options:
  -i, --input-path <INPUT>    Input image path
  -o, --output-path <OUTPUT>  Output image path
  -C, --use-cpu               Use CPU instead of GPU for inference
  -m, --model-name <MODEL>    Filename of the model in `models` directory [default: pro-no-denoise-up2x.pth]
  -a, --alpha <ALPHA>         Please check the documentation for this option [default: 1.0]
  -t, --tile-size <TILE>      Tile size, smaller value can reduce memory usage
      --no-cache              Disable cache
  -h, --help                  Print help
  -V, --version               Print version
```

支持的图片格式：BMP、JPEG、PNG、WebP。

### 注意事项

- 目前仅支持 pro 模型的 2 倍超分。
- **欢迎 PR！**
