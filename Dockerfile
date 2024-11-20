# ベースイメージとして RAPIDS の公式イメージを使用
FROM rapidsai/notebooks:24.08-cuda11.8-py3.10

# 必要なライブラリをインストール
RUN pip install --no-cache-dir tensorflow==2.10.1 \
    torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1

# 作業ディレクトリの設定
WORKDIR /home/rapids

# JupyterLab の起動コマンド
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
