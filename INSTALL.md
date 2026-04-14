## Installation

### Requirements
- Linux with Python ≥ 3.6
- PyTorch ≥ 1.8 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).

### Usage

This repository can be managed with `uv` without calling `pip` directly.

#### Base environment with `uv`

```bash
uv python install 3.9
uv sync --python 3.9
```

Optional components are exposed as extras:

```bash
# local third_party/CLIP in editable mode
uv sync --python 3.9 --extra clip

# local open_clip_training package in editable mode
uv sync --python 3.9 --extra open-clip

# web demo dependencies
uv sync --python 3.9 --extra demo

# test dependencies
uv sync --python 3.9 --extra test
```

#### Detectron2 note

The original project points to the Detectron2 0.6 CUDA 11.3 wheel index:

```text
https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```

That index only provides Linux wheels for `cp36` to `cp39`; it does not ship a
`cp310` wheel. Also, `torch==1.10.1` itself only provides wheels up to `cp39`.

For the original OVSeg stack (`torch==1.10.1`, `torchvision==0.11.2`,
`detectron2==0.6`), use Python 3.9. This repository now pins the matching
prebuilt wheels directly in `pyproject.toml`, so a plain sync is enough:

```bash
uv python install 3.9
uv sync --python 3.9
```

Using the Git source build for Detectron2 on a modern machine often fails
because the local CUDA toolkit is newer than the CUDA version used to build the
old PyTorch 1.10 wheels. The prebuilt wheel path avoids that mismatch.

#### Legacy reference

The original README used the following `conda`/`pip` flow:

```bash
conda create --name ovseg python=3.8
conda activate ovseg
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

You need to download `detectron2==0.6` following [instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

```bash
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```


FurtherMore, install the modified clip package.

```bash
cd third_party/CLIP
python -m pip install -Ue .
```
