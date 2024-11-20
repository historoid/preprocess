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

### カスタムイメージのビルド

Dockerがインストールされていることが前提です。  
Dockerfileからローカルにイメージを作成するか、DockerHubからpullしてください。

```bash
# ビルドする場合
# Dockerfileが必要です
docker build -t historoid/rapids-notebook:latest .

# DockerHubから持ってくる場合
docker login
docker pull historoid/rapids-notebook:latest
```

これでlatestバージョンがローカルに保存されます。  
もしこのローカルバージョンに修正を加えたうえで、別のバージョン名としてDockerHubにアップロードしたい場合は以下のコマンドを実行してください。

```bash
# 新しいイメージを作成する
# はじめにコンテナIDを確認する
docker ps -a

# そのコンテナIDで新しくイメージを作成
# docker commit abc123 historoid/rapids-notebook:win
docker commit container_id user_name/image_name:version_name

# upload
# docker login
docker push historoid/rapids-notebook:win

# latest が実際どのバージョンなのかを明示する
docker tag historoid/rapids-notebook:win historoid/rapids-notebook:latest
docker push historoid/rapids-notebook:latest
```

### GPUを使えているかどうかの確認

```python
# PyTorchの場合
import torch
torch.backends.mps.is_available()

# TensorFlowの場合
import tensorflow as tf
tf.config.list_physical_devices('GPU')
```