# Independent flood-extent extraction

This package is separate from `compare/geometry` and the flood-depth training code.
It trains one flood-extent extractor, freezes its checkpoint, and exports one
reusable georeferenced product for FwDET, RICorDE local-HAND, and FLEXTH.

## Label and objective

By experiment definition, `valid_depth_mask` is the binary flood label:

- `1`: flood;
- `0`: non-flood inside `output_valid`;
- pixels outside `output_valid`: ignored.

Training minimizes pure masked Soft-IoU loss. The best checkpoint maximizes
validation raw pixel-micro IoU at the fixed probability threshold 0.5. Precision,
recall and F1 are reported from the same binary label. The 80 m buffered product is
also evaluated, but buffered IoU does not select the checkpoint.

The architecture follows the Sentinel-1 early-fusion change-detection route in
Misra et al., *Nature Communications* (2025): a MobileNetV2 encoder, U-Net decoder,
and pre/event VV/VH threshold-change inputs. The public thresholds are converted
back from its scaled representation to -17.5/-22.5 dB water thresholds, a 5 dB
minimum drop, and -30/-32.5 dB validity floors. The fixed 80 m buffer is four local
20 m pixels.

## Frozen run

- parameters: 6,628,817;
- best epoch: 8;
- validation raw IoU / F1: 0.342388 / 0.510117;
- test raw IoU / F1: 0.359321 / 0.528677;
- test raw precision / recall: 0.365170 / 0.957320;
- test buffered IoU / F1: 0.295989 / 0.456777;
- checkpoint SHA256: `cefab47004ee772743406254a48afaff624afb385687e8e2bca743b28aa0da28`.

Training artifacts are under `runs/extent/train/subset150_ai4g_mobilenet_iou_frozen`.
The frozen val/test extent product is under
`runs/extent/products/subset150_ai4g_mobilenet_iou_frozen`. Flood-depth outputs are
not written inside either directory.

Primary sources: [paper](https://www.nature.com/articles/s41467-025-60973-1),
[official implementation](https://github.com/microsoft/ai4g-flood), and
[Kuro Siwo benchmark](https://proceedings.neurips.cc/paper_files/paper/2024/hash/43612b0662cb6a4986edf859fd6ebafe-Abstract-Datasets_and_Benchmarks_Track.html).
