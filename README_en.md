# mmpose-webcam-demo

[简体中文](/README.md) | English

A template to build demos and applications with MMPose Webcam API

<div align="center">
    <img src="https://user-images.githubusercontent.com/15977946/171618680-49968673-6f11-4b9d-b63e-72543e8a75a0.gif">
</div>

## Configure Environment

### Create a conda environment

```shell
conda create -n mmpose-demo python=3.9 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate mmpose-demo
```

### Install MMCV and MMDetection

```shell
pip install openmim
mim install mmcv-full
pip install mmdet
```

### Install MMPose

Clone MMPose and install it in the editable mode, so the latest updates of MMPose can be synchronized easily.

```shell
cd ..
git clone clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -e .
```

Check MMPose installation.

```shell
python -c "from mmpose.apis import webcam"
```

### Configure pre-commit hook

```shell
# In mmpose-webcam-demo repo folder
pip install pre-commit
pre-commit install
```

## Run the Demo

```shell
python run.py --configs configs/pose_estimation/pose_estimation.py
```

## Useful Linnks

- Webcam API
  - [Tutorial](https://mmpose.readthedocs.io/en/latest/tutorials/7_webcam_api.html)
  - [API Reference](https://mmpose.readthedocs.io/en/latest/api.html#mmpose-apis-webcam)
- MMPose
  - [Code](https://github.com/open-mmlab/mmpose)
  - [Documentation](https://mmpose.readthedocs.io/en/latest/)
  - [Model Zoo](https://mmpose.readthedocs.io/en/latest/modelzoo.html)
- About "Infinity Pose" MMPose Creative Demo Competition
  - [Event Home Page](https://openmmlab.com/community/mmpose-demo)
  - [Submission](https://github.com/open-mmlab/mmpose/issues/1407)
