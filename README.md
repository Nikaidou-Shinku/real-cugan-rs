## Real-CUGAN-rs

A Rust port of [Real-CUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN).

### Usages

Short view:

```shell
real-cugan-rs -i input.webp -o output.webp
```

Full help text:

```console
A Rust port of Real-CUGAN

Usage: real-cugan-rs [OPTIONS] --input-path <INPUT> --output-path <OUTPUT>

Options:
  -i, --input-path <INPUT>    Input image path
  -o, --output-path <OUTPUT>  Output image path
  -c, --use-cpu               Use CPU instead of GPU for inference
  -m, --model-name <MODEL>    Filename of the model in `models` directory [default: pro-no-denoise-up2x.pth]
  -h, --help                  Print help
  -V, --version               Print version
```

Supported image formats: JPEG, PNG, WebP.

### Note

- Currently only 2x upscale of the pro model is supported.
- Currently no tile support.
- Currently transparent images are not supported.
- **PRs are welcome!**
