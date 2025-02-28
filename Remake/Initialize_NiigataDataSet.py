import os
from rich.panel import Panel
from rich.console import Console

from myfunctions import files as mf

console = Console()

settings_path = "./settings.yaml"
settings = mf.load_settings(settings_path)

# オリジナルフォルダのコピー
try:
    mf.copy_directory_with_metadata(settings['root_dir'], settings['backup_dir'], overwrite=True)
    message = f"Successfully copied '{settings['root_dir']}' to '{settings['backup_dir']}'."
    console.print(Panel(message, style="bold green", expand=False))
except Exception as e:
    console.print(Panel(f"Error during copy: {e}", style="bold red", expand=False))
    exit()

# 不要なファイル・フォルダの削除
try:
    mf.remove_unnecessary_data(settings['backup_dir'], set(settings['unwanted_files']))
    message = f"Successfully cleaned up files and folders in '{settings['backup_dir']}'."
    console.print(Panel(message, style="bold green", expand=False))
except Exception as e:
    console.print(Panel(f"Error during cleanup: {e}", style="bold red", expand=False))

# ファイル名・フォルダ名の正規化
try:
    renamed_entries, error_entries = mf.rename_and_normalize(settings['backup_dir']) # 修正
    if renamed_entries:
        message = f"Successfully renamed files and directories in '{settings['backup_dir']}'."
        console.print(Panel(message, style="bold green", expand=False))
    if error_entries:  # エラーがあれば表示
        console.print(Panel("[red]Errors occurred during renaming:[/]", title="[bold red]Errors"))
        for path, error in error_entries:
            console.print(f"[red]- {path}:[/] {error}")
except Exception as e:
     console.print(Panel(f"Error during rename: {e}", style="bold red", expand=False))

# 単独ファイルの整理
allowed_exts = set([ext.lower() for ext in settings['image_ext'] + settings['office_ext']])
try:
    mf.organize_patient_data(settings['backup_dir'], settings['patient_root'], allowed_exts)
except Exception as e:
    console.print(Panel(f"Error during organizing standalone files: {e}", style="bold red", expand=False))
