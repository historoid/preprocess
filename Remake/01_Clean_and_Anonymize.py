import os
import yaml
import logging
from rich.panel import Panel
from rich.console import Console

from myfunctions import files as mf

# Richコンソールの初期化
console = Console()

# ログディレクトリの作成（なければ）
log_dir = "./logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# logging の基本設定
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename=os.path.join(log_dir, "dataset_processing.log"),
    filemode="w"
)

# コンソールにも INFO 以上を出力するハンドラーを追加
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(formatter)
logging.getLogger("").addHandler(console_handler)

# 設定ファイルの読み込み
settings_path = "./settings.yaml"
with open(settings_path, "r", encoding="utf-8") as f:
    settings = yaml.safe_load(f)

# ルートディレクトリ（バックアップディレクトリ）のパス
root_dir = settings["backup_dir"]

# ルートフォルダの存在確認
if not os.path.exists(root_dir):
    logging.critical(f"Root folder '{root_dir}' does not exist.")
    console.print(Panel(f"Root folder '{root_dir}' does not exist.", style="bold red"))
    exit(1)
else:
    logging.info(f"Root folder '{root_dir}' exists.")
    console.print(Panel(f"Root folder '{root_dir}' exists.", style="bold green"))

# 患者フォルダ直下以外に例外的なフォルダが存在するかチェック
all_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
non_flat_dirs = []  # サブディレクトリが含まれている患者フォルダ
empty_dirs = []     # ファイルが全く存在しない患者フォルダ

for d in all_dirs:
    patient_path = os.path.join(root_dir, d)
    entries = os.listdir(patient_path)
    # 患者フォルダ内に少なくとも1つのファイルが存在するかチェック
    if not any(os.path.isfile(os.path.join(patient_path, entry)) for entry in entries):
        empty_dirs.append(patient_path)
    # 患者フォルダ内にサブディレクトリが存在するかチェック
    if any(os.path.isdir(os.path.join(patient_path, entry)) for entry in entries):
        non_flat_dirs.append(patient_path)

if empty_dirs or non_flat_dirs:
    message = "以下の患者フォルダに例外的な構造が検出されました："
    if empty_dirs:
        message += "\n■ 空のフォルダ（ファイルが存在しません）:\n" + "\n".join(empty_dirs)
    if non_flat_dirs:
        message += "\n■ サブディレクトリが含まれているフォルダ:\n" + "\n".join(non_flat_dirs)
    logging.warning(message)
    console.print(Panel(message, style="bold yellow"))
else:
    logging.info("すべての患者フォルダは正しく構造化されています（空でなく、フラットな構造）。")
    console.print(Panel("すべての患者フォルダは正しく構造化されています（空でなく、フラットな構造）。", style="bold green"))

# 患者フォルダ（症例）の数を確認（ルート直下のディレクトリ数）
patient_dirs = [d for d in all_dirs if os.path.isdir(os.path.join(root_dir, d))]
case_count = len(patient_dirs)
logging.info(f"Total number of cases (directories) in root folder: {case_count}")
console.print(Panel(f"Total number of cases (directories) in root folder: {case_count}", style="bold green"))

# ルートフォルダ内のすべてのファイルの拡張子を確認（再帰的に）
all_extensions = set()
for dirpath, dirnames, filenames in os.walk(root_dir):
    for f in filenames:
        ext = os.path.splitext(f)[1].lower()
        all_extensions.add(ext)
logging.info(f"Unique file extensions found: {all_extensions}")
console.print(Panel(f"Unique file extensions found: {all_extensions}", style="bold green"))

# 不要なファイル・フォルダの削除
try:
    mf.remove_unnecessary_data(
        settings['backup_dir'],
        set(settings['unwanted_files']),
        set(settings['image_ext'])
    )
    message = f"Successfully cleaned up files and folders in '{settings['backup_dir']}'."
    logging.info(message)
    console.print(Panel(message, style="bold green", expand=False))
except Exception as e:
    err_msg = f"Error during cleanup: {e}"
    logging.error(err_msg)
    console.print(Panel(err_msg, style="bold red", expand=False))
