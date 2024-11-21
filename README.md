# macOSバージョン（dev/macブランチ）について

このブランチでは、M3 Macを使った開発を記録します。

Apple SiliconでGPUを利用するには、Metal APIを介する必要があるため、Dockerは利用できません。  
そこでcondaでの環境管理を行います。

## JupyterLabについて

実際の開発はJupyterLabを介して行います。  
jupyterでもcondaの仮想環境を認識するように設定を行ってください。

## 開発環境の作り方について

### miniforgeによる仮想環境構築の流れ

まずはHomebrewで、miniforgeをインストールしてください。

```bash
# miniforgeのインストール
brew install miniforge

# ターミナルの再起動
conda init zsh  # ここでターミナルを再起動する

# ベース環境のアクティベート
conda activate base  # プロンプトに(base)と表示される

# 仮想環境を作成する
conda env create -f environment.yml

# 仮想環境を立ち上げる
conda activate ml_env  # プロンプトが変わる

# 仮想環境をダウンする
conda deactivate
```
### 仮想環境をjupyterlabに認識させる

コーディングは、Jupyterlab上で行う予定であるが、Jupyter上でライブラリをインストールすると、base環境にインストールされてしまう。  
そのため仮想環境があること自体をjupyter側に教えておく必要がある。

```bash
# 必要があれば
pip install ipykernel  # environment.ymlでインストール済みのはず

python  -m ipykernel install --user --name=env_name --display-name "Python(env_name)"
```
* --name: 仮想環境の名前
* --display-name: JupterLab上で表示されるカーネル名

###  environment.yml

environment.ymlファイルはこのブランチで管理する。

### GPUを使えているかどうかの確認

```python
# PyTorchの場合
import torch
torch.backends.mps.is_available()

# TensorFlowの場合
import tensorflow as tf
tf.config.list_physical_devices('GPU')
```

### JupyteLabの設定変更

```bash
jupyter lab --generate-config  # ~/.jupyter/jupyter_lab_config.pyが作られる
```

```python
# JupyterLabを全IPアドレスでリッスン
c.ServerApp.ip = '0.0.0.0'

# 自動的にブラウザを開かない
c.ServerApp.open_browser = False

# ルートユーザーでの実行を許可
c.ServerApp.allow_root = True
```