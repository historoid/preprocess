import os
import yaml
import shutil
from typing import Dict
from typing import Union


def load_settings(yaml_path: str) -> Dict:
    """
    Load settings from a YAML file and define global variables for each key-value pair.
    
    Args:
        yaml_path (str): Path to the YAML file.

    Returns:
        dict: The settings loaded from the YAML file.
    """
    # パスの型チェック
    assert isinstance(yaml_path, str), "YAML path must be a string."
    
    # ファイル存在確認
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Settings file '{yaml_path}' does not exist.")
    
    # YAMLファイルを読み込む
    with open(yaml_path, 'r', encoding='utf-8') as file:
        settings = yaml.safe_load(file)
    
    return settings



def copy_directory_with_metadata(src: str, dst: str) -> None:
    """
    Copies all contents of the source directory to the destination directory,
    preserving metadata (e.g., timestamps).
    
    Args:
        src (str): Path to the source directory.
        dst (str): Path to the destination directory.

    Raises:
        FileNotFoundError: If the source directory does not exist.
        FileExistsError: If the destination directory already exists.
    """
    # 型チェック
    assert isinstance(src, str) and isinstance(dst, str), "Source and destination paths must be strings."
    
    # ソースディレクトリの存在確認
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source directory '{src}' does not exist.")
    
    # デスティネーションディレクトリの存在確認
    if os.path.exists(dst):
        raise FileExistsError(f"Destination directory '{dst}' already exists.")

    # ルートディレクトリの作成
    os.makedirs(dst, exist_ok=True)

    # ディレクトリの走査とコピー
    for root, dirs, files in os.walk(src):
        # 現在のルートの相対パスを計算
        relative_path = os.path.relpath(root, src)
        dst_root = os.path.join(dst, relative_path)
        
        # ディレクトリのコピー
        for dir_name in dirs:
            os.makedirs(os.path.join(dst_root, dir_name), exist_ok=True)
        
        # ファイルのコピー
        for file_name in files:
            src_file = os.path.join(root, file_name)
            dst_file = os.path.join(dst_root, file_name)
            shutil.copy2(src_file, dst_file)  # メタデータを保持するためcopy2を使用
