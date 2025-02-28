import os
from rich.panel import Panel
from rich.console import Console

from myfunctions import files as mf

console = Console()

settings_path = "./settings.yaml"
settings = mf.load_settings(settings_path)

# オリジナルフォルダのコピー
try:
    mf.copy_directory_with_metadata(settings['root_dir'], settings['backup_dir'],True)
    message = f"Successfully copied '{settings['root_dir']}' to '{settings['backup_dir']}'."
    console.print(Panel(message, style="bold green", expand=False))
except Exception as e:
    print(f"Error: {e}")

# 不要なファイル・フォルダの削除