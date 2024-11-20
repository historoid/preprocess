# このpreprocessリポジトリについて

## Windowsバージョン（dev/winブランチ）について

このブランチでは、NVIDIA GPUを搭載したWindows11を使った開発を記録します。

WSL2上にDockerコンテナを作って開発環境を作成します。

## JupyterLabについて

pythonのコーディングはJupyterLab上で行います。

## 開発環境の作り方について

### docker-compose.yml

このリポジトリは主にdocker-compose.ymlファイルを管理するためのものです。

ここにコンテナの作成に関する記述をします。

### GPUを使えているかどうかの確認

```python
# PyTorchの場合
import torch
torch.backends.mps.is_available()

# TensorFlowの場合
import tensorflow as tf
tf.config.list_physical_devices('GPU')
```