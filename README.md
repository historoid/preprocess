# このpreprocessリポジトリについて

このリポジトリは、災害犠牲者身元確認（Diseaster Victim Identification）プロジェクトの一部です。  
口腔内写真（intra-oral photos）をデータセットとして、機械学習によって口腔内所見を自動的に認識することが目的です。

## データセットについて

データセットは非公開です。  
.gitignoreファイルで画像ファイルはGitHub上にアップロードされないように設定してあります。

## preprocess（前処理）について

収集されたデータは、不要な画像やドキュメントも含まれています。  
これらを取り除いたり、あまりに類似した画像を除外することが本リポジトリの目的です。

## ブランチについて

macOS, Windowsの双方を利用できるようにしています。  
開発や修正は、dev/mac, dev/win のリポジトリを使います。  

### macOSバージョンについて

macOS版は、Apple SilliconのMacを対象としています（作者の環境はM3）。  
前処理においてもGPUアクセラレーションを利用するために、TensorFlow, PyTorch を直接macOSにインストールします。  
開発環境の管理のためにminiforge（condaの軽量版）を利用します。

miniforgeで利用しやすいように、environment.ymlファイルに諸々の環境設定を記述します。

### Windowsバージョンについて

作者は、windows11を使っており、WindowsマシンにはNVIDIAのGPUが搭載されています。  
そのため、Windows版ではcudaを前提として環境設定を行います。

Windows板では、WSL2 + Docker を利用して開発環境を管理します。  
NVIDIAが公開しているコンテナを利用します。

## macOSバージョン（dev/macブランチ）について

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
conda activate env_name  # プロンプトが変わる

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