import os
from rich.panel import Panel
from rich.console import Console

from myfunctions import files as mf

console = Console()

settings_path = "./settings.yaml"
settings = mf.load_settings(settings_path)

# オリジナルフォルダのコピー
try:
    mf.copy_directory_with_metadata(settings['root_dir'], settings['backup_dir'], True)
    message = f"Successfully copied '{settings['root_dir']}' to '{settings['backup_dir']}'."
    console.print(Panel(message, style="bold green", expand=False))
except Exception as e:
    console.print(Panel(f"Error during copy: {e}", style="bold red", expand=False))
    exit()

# 不要なファイル・フォルダの削除
unwanted_file_names = {
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
    "ehthumbs.db",
    "ehthumbs_vista.db",
    "$RECYCLE",
    ".Trach",
    ".localized",
    ".thumbnails"
}
allowed_extensions = set([ext.lower() for ext in settings['image_ext'] + settings['office_ext']])

try:
    mf.remove_unnecessary_data(settings['backup_dir'], allowed_extensions, unwanted_file_names)
    message = f"Successfully cleaned up files and folders in '{settings['backup_dir']}'."
    console.print(Panel(message, style="bold green", expand=False))

except Exception as e:
    console.print(Panel(f"Error during cleanup: {e}", style="bold red", expand=False))
