# LineFormer (fork): the same model, runnable on AMD ROCm and faster in batches

This is a fork of [TheJaeLal/LineFormer](https://github.com/TheJaeLal/LineFormer), the official code of
*LineFormer: Line Chart Data Extraction Using Instance Segmentation* (Lal et al., ICDAR 2023). It runs the same
model with the same checkpoint and gives the same results (checked image by image against the original stack, see
[docs/VALIDATION.md](docs/VALIDATION.md)). What it adds:

* **Compatibility**: inference on GPUs without mmcv's compiled ops, e.g. AMD GPUs with ROCm, through mmcv's own
  pure-PyTorch MultiScaleDeformableAttention (`msda_compat.py`); nothing to compile: the pure-Python part of mmcv
  1.7.2 that inference uses is vendored in `third_party/mmcv` (compiled ops are stand-ins that raise if called);
  an install script for a current stack (Python 3.13, torch 2.14 ROCm 7.2, numpy 2.5, OpenCV 5.0).
* **Speed**: a kept-queries mode that post-processes only the queries that can reach the 0.3 threshold
  (`kept_queries.py`; same lines, detector time per image 0.27 s -> 0.056 s, peak device memory ~12 GB -> 0.5 GB),
  and a job engine
  (`lineformer batch`, `lineformer serve`, a stdlib client) that keeps the GPU busy with parallel pre- and
  post-processing (~20 images/s on one RX 7900 XTX, against 5.8 s per image on the original CPU stack).
* A `lineformer` command, unit tests that run without GPU or checkpoint, and the equivalence harness used for the
  validation (`tools/equivalence/`).

This fork is not affiliated with the authors of LineFormer. If you use it, please cite their paper (see
[Citation](#citation) below and `CITATION.cff`).

## Quick start

Linux or WSL2 with an AMD GPU (ROCm 7.2 in `/opt/rocm`), `git` and [uv](https://docs.astral.sh/uv/); no compiler:

```bash
git clone https://github.com/jehelmich/LineFormer.git ~/LineFormer && cd ~/LineFormer
VENV=$HOME/lineformer bash rocm/install_rocm.sh       # downloads torch; nothing is compiled
# download iter_3000.pth from the authors' link under "Inference" below
$HOME/lineformer/bin/lineformer --ckpt iter_3000.pth --out /tmp/lf_demo demo/PMC5959982___3_HTML.jpg
# expect /tmp/lf_demo/PMC5959982___3_HTML.json with 3 lines of 621 points
```

A plain `pip install git+...` is not enough: torch comes from its own index, and the vendored mmcv subset and
mmdetection are installed from the checkout. CPU only or NVIDIA: the same script with another torch index
(`rocm/INSTALL.md`; NVIDIA not tested by this fork). Many images:

```bash
lineformer batch --ckpt iter_3000.pth --list images.txt --out out/ --gpu-workers 2   # one job, all cores
lineformer serve --ckpt iter_3000.pth --port 8775 --gpu-workers 2                    # a server that owns the GPU
```

Defaults: `--device auto` (GPU if PyTorch sees one, else CPU), kept-queries mode on at 0.3 (`--all-queries` for all
100 instances per image, as upstream). `--input-size native` and `--tile` are experimental (see
[docs/VALIDATION.md](docs/VALIDATION.md#input-scale)). Tests: `python -m pytest` or `python tests/run_all.py`;
lint: `ruff check` (the fork's own files only).

`lineformer batch` and `lineformer serve` write the lines and instances in a deterministic geometric order (the
model's own order differs between CPU and GPU): lines by leftmost x, then mean y, then score; line *i* is instance
*i* of `<id>.instances.npz` / `<id>.masks.npz`, and the instances without a line follow (details in
`lineformer_jobs.py`). `infer.get_dataseries` and the single-process `lineformer` keep the model's order, as
upstream. Every job writes a manifest `<out>/job.json` (fork version and git commit, package versions, options,
per-image status and output sha256, start and end times, failures).

* [rocm/INSTALL.md](rocm/INSTALL.md): install, update, environment check, use (batch, serve, client, Python)
* [docs/VALIDATION.md](docs/VALIDATION.md): how equivalence and speed were measured, and the limits
* [CHANGELOG.md](CHANGELOG.md): changes against upstream

## Licensing

* The upstream LineFormer code and the checkpoint carry no licence from their authors, so all rights are reserved by
  default (see upstream issue [#12](https://github.com/TheJaeLal/LineFormer/issues/12)). Users who need clear rights
  should contact the authors. This fork therefore has no top-level LICENSE file.
* The vendored `mmdetection/` and `third_party/mmcv/` (a subset of mmcv 1.7.2; changes in its `NOTICE.md`) are
  Apache-2.0 (OpenMMLab, see their `LICENSE`).
* The files added in this fork are Apache-2.0 ([LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt); each carries an
  SPDX header). The modifications to upstream files (`infer.py`, `README.md`, `.gitignore`) are contributed under
  Apache-2.0 as far as they are separable from the upstream code.

---

*The authors' original README follows, unchanged.*

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
