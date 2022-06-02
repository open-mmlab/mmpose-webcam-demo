# mmpose-webcam-demo

A template to build demos and applications with MMPose Webcam API

<div align="center">
    <img src="https://user-images.githubusercontent.com/15977946/171618680-49968673-6f11-4b9d-b63e-72543e8a75a0.gif">
</div>

## 配置环境

### 创建虚拟环境

```shell
conda create -n mmpose-demo python=3.9 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate mmpose-demo
```

### 安装 MMCV 和 MMDetection

```shell
pip install openmim
mim install mmcv-full
pip install mmdet
```

### 安装 MMPose

为了能随时同步最新的 MMPose 代码，我们推荐将 MMPose 克隆到本地，并通过开发模式安装

```shell
cd ..
git clone clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -e .
```

测试 MMPose 安装成功

```shell
python -c "from mmpose.apis import webcam"
```

### 配置 pre-commit hook

```shell
# 在 mmpose-webcam-demo 目录中执行以下操作
pip install pre-commit
pre-commit install
```

## 运行示例

```shell
python run.py --configs configs/pose_estimation/pose_estimation.py
```

## 相关链接

- [MMPose 代码仓库](https://github.com/open-mmlab/mmpose)
- [MMPose 官方文档](https://mmpose.readthedocs.io/en/latest/)
- [MMPose Webcam API 文档](https://mmpose.readthedocs.io/en/latest/api.html#mmpose-apis-webcam)
- [MMPose Model Zoo](https://mmpose.readthedocs.io/en/v0.26.0/modelzoo.html)
