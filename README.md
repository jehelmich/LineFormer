# LineFormer - Rethinking Chart Data Extraction as Instance Segmentation,
Jay Lal, Aditya Mitkari*, [Mahesh Bhosale*](https://bhosalems.github.io/), David Doermann, International Conference on Document Analysis and Recognition, 2023.

Official repository for the ICDAR 2023 Paper

[<u>[Link]</u>](https://link.springer.com/chapter/10.1007/978-3-031-41734-4_24) to the paper.

## Quantitative Results
| Dataset             | AdobeSynth19 Visual Element Detection[^1] | Data Extraction[^2] | UB-PMC22 Visual Element Detection | Data Extraction | LineEX Visual Element Detection | Data Extraction |
|---------------------|------------------------------------------|---------------------|----------------------------------|-----------------|---------------------------------|----------------|
| [ChartOCR](https://openaccess.thecvf.com/content/WACV2021/papers/Luo_ChartOCR_Data_Extraction_From_Charts_Images_via_a_Deep_Hybrid_WACV_2021_paper.pdf)        | 84.67                                    | 55                  | 83.89                            | 72.9            | 86.47                           | 78.25          |
| [Lenovo](https://link.springer.com/chapter/10.1007/978-3-030-86549-8_37)          | **99.29**                                | **98.81**          | 84.03                            | 67.01           | -                               | -              |
| [LineEX](https://openaccess.thecvf.com/content/WACV2023/papers/P._LineEX_Data_Extraction_From_Scientific_Line_Charts_WACV_2023_paper.pdf)          | 82.52                                    | 81.97               | 50.23                         | 47.03           | 71.13                           | 71.08          |
| [**Lineformer**](https://arxiv.org/abs/2305.01837) (Ours)   | 97.51                                    | 97.02               | **93.1**                          | **88.25**       | **99.20**                       | **97.57**      |

[^1]: [task-6a from CHART-Info challenge](https://example.com/chart-info-task-6a)
[^2]: [task-6b data score from CHART-Info challenge](https://example.com/chart-info-task-6b)

<!-- **If you would like to cite our work:**
```latex

``` -->

## Model Usage
### Install Environment

This code is based on [MMdetection Framework](https://github.com/open-mmlab/mmdetection).

Code has been tested on Pytorch 1.13.1 and CUDA 11.7.

Create Conda Environment and install dependencies:
```bash
conda create -n LineFormer python=3.8
conda activate LineFormer
bash install.sh
```


### Inference

1. Download the Trained Model Checkpoint [here](https://drive.google.com/drive/folders/1K_zLZwgoUIAJtfjwfCU5Nv33k17R0O5T?usp=sharing)
2. Use the demo inference snippet shown below

```python
import infer
import cv2
import line_utils

img_path = "demo/PMC5959982___3_HTML.jpg"
img = cv2.imread(img_path) # BGR format

CKPT = "iter_3000.pth"
CONFIG = "lineformer_swin_t_config.py"
DEVICE = "cpu"

infer.load_model(CONFIG, CKPT, DEVICE)
line_dataseries = infer.get_dataseries(img, to_clean=False)

# Visualize extracted line keypoints
img = line_utils.draw_lines(img, line_utils.points_to_array(line_dataseries))
    
cv2.imwrite('demo/sample_result.png', img)


```

Example extraction result:

![input image](demo/PMC5959982___3_HTML.jpg "Input")
![demo result](demo/sample_result.png "Detection Result")

### Running on AMD ROCm / without compiled GPU ops

The only compiled mmcv op LineFormer needs on a GPU is `MultiScaleDeformableAttention` (pixel decoder).
mmcv also ships the same algorithm in pure PyTorch (`multi_scale_deformable_attn_pytorch`, the path it always
takes for CPU tensors). `infer.load_model` takes an optional `msda` argument (or the environment variable
`LINEFORMER_MSDA`) that chooses the path on a GPU:

* `auto` (default): mmcv's compiled kernel if a tiny call through it runs on the device, else the pure-PyTorch one;
* `compiled`: the compiled kernel; raises if mmcv was built without it;
* `pytorch`: the pure-PyTorch implementation.

The chosen path is printed once. On `cpu` nothing changes. Thresholds, preprocessing and postprocessing are
the same on every device; the order of the returned instances (and so of the lines from `get_dataseries`) can
differ between CPU and GPU.

```python
infer.load_model(CONFIG, CKPT, "cuda", msda="auto")  # "cuda" / "cuda:0" also selects an AMD GPU with ROCm PyTorch
```

Tested setup: AMD Radeon RX 7900 XTX (gfx1100), ROCm 7.2.0, WSL2 Ubuntu 24.04, Python 3.11.17,
torch 2.14.1+rocm7.2, torchvision 0.29.1+rocm7.2 (download.pytorch.org/whl/rocm7.2), mmcv-full 1.7.2 built
from source with CPU ops only, mmdet 2.28.2 (vendored), numpy 1.23.5, opencv-python 4.11.0.86, scipy 1.9.3,
scikit-image 0.21.0. `rocm/install_rocm.sh` builds this environment; `rocm/mmcv-1.7.2-cpu-ops.patch` makes mmcv's
`setup.py` compile with C++20 (needed by the headers of torch >= 2.10) and skip its CUDA/HIP auto-detection when
`MMCV_CPU_ONLY=1`. Under WSL the wheel's `libhsa-runtime64.so` is replaced by the one from `/opt/rocm`.

Verified on 72 chart images against the original stack (Python 3.8, torch 1.13.1 CPU, mmcv-full 1.7.2 with
compiled ops): the same instances above the 0.3 threshold on every image, mask IoU >= 0.9999, scores within
1.1e-5, every line point within 1 px (`tools/equivalence/` holds the harness).

See `rocm/INSTALL.md` for install, update and dependency rules.

#### Kept-queries mode (opt-in speed-up)

mmdet upsamples all 100 query masks to the original image size, scores and copies every one, although only a few
reach LineFormer's 0.3 threshold. The final score is class score x mask score with mask score <= 1, so a query whose
class score is below 0.3 cannot reach 0.3. `kept_queries.py` drops those queries before the upsample and does the
rest on the device for the kept ones only (same mmdet operations), with one host copy for their masks:

```python
infer.load_model(CONFIG, CKPT, "cuda", kept_only=True)  # or kept_only=0.25; env LINEFORMER_KEPT_QUERIES=on|<thr>
```

`lineformer --kept-only [--kept-thr 0.3]` on the command line. Default: off (mmdet untouched). **Instances whose class
score is below the threshold are not returned** (the box array has one row per kept query, a few of which can still
have a final score below the threshold); code that reads low-scoring instances must leave the mode off. It is refused
with an error unless mmdet is 2.28.2 with unchanged source of the reproduced functions, and the model has one thing
class and no stuff class.

Measured on the same 72 images on the RX 7900 XTX: the kept masks are bit-identical to the unpatched path (126 of 126
instances), scores within 2.3e-6 (the unpatched GPU path itself varies by up to 1.4e-6 between runs), dataseries
identical, and the acceptance against the original stack passes on 72 of 72 images. Detector time per image 0.27 s ->
0.056 s, peak device memory per process 12 GB -> 0.5 GB; with `tools/throughput/batch_infer.py` 2.5 -> 5.3 images/s
in one process, 3.4 -> 18.7 images/s pipelined (one GPU worker), 20.6 images/s with two GPU workers.

#### Command line

`pip install --no-deps -e .` (done by `rocm/install_rocm.sh`) installs a `lineformer` command:

```bash
lineformer --ckpt iter_3000.pth --device cuda:0 --out out/ chart1.png chart2.png
lineformer --ckpt iter_3000.pth --list images.txt --out out/ --masks   # one path per line; also the kept masks
```

It writes `<out>/<stem>.json` (`{"image": ..., "lines": [[{"x":..,"y":..}, ...], ...]}`, from `get_dataseries`)
and with `--masks` `<stem>.masks.npz`; existing outputs are skipped unless `--force`. `--kept-only` switches on the
kept-queries mode above (same lines, about 4x less GPU time per image).

Please cite the LineFormer paper (see [Citation](#citation) and `CITATION.cff`) when you use this code.

## Citation
If you found our work useful, please cite us as follows:
```bib
@InProceedings{10.1007/978-3-031-41734-4_24,
author="Lal, Jay
and Mitkari, Aditya
and Bhosale, Mahesh
and Doermann, David",
editor="Fink, Gernot A.
and Jain, Rajiv
and Kise, Koichi
and Zanibbi, Richard",
title="LineFormer: Line Chart Data Extraction Using Instance Segmentation",
booktitle="Document Analysis and Recognition - ICDAR 2023",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="387--400",
abstract="Data extraction from line-chart images is an essential component of the automated document understanding process, as line charts are a ubiquitous data visualization format. However, the amount of visual and structural variations in multi-line graphs makes them particularly challenging for automated parsing. Existing works, however, are not robust to all these variations, either taking an all-chart unified approach or relying on auxiliary information such as legends for line data extraction. In this work, we propose LineFormer, a robust approach to line data extraction using instance segmentation. We achieve state-of-the-art performance on several benchmark synthetic and real chart datasets. Our implementation is available at https://github.com/TheJaeLal/LineFormer.",
isbn="978-3-031-41734-4"
}
```

## Full Plot Data Extraction
Note: LineFormer returns data in form of x,y points w.r.t the image, to extract full data-values you need to extract axis information. 
Please refer the following resources:
* [E2E Line Chart Data extraction](https://github.com/tdsone/extract-line-chart-data) implementation put together by [@tdsone](https://github.com/tdsone)
* Chart Element Detection [this](https://github.com/pengyu965/ChartDete/) repo.
