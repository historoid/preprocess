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

### JupyterLabについて

実際の開発はJupyterLabを介して行います。  
jupyterでもcondaの仮想環境を認識するように設定を行ってください。