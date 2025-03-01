import os
from rich.panel import Panel
from rich.console import Console

from myfunctions import files as mf
from myfunctions import office as mo

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
    mf.remove_unnecessary_data(
        settings['backup_dir'],
        set(settings['unwanted_files']),
        set(setings['image_ext'] + settings['office_ext'])
    )
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


# 旧オフィスファイルの変換処理（.doc, .xls, .ppt → .docx, .xlsx, .pptx）
try:
    mo.convert_office_files(settings['backup_dir'])
    console.print(Panel("Officeファイルの変換が完了しました。", style="bold green", expand=False))
except Exception as e:
    console.print(Panel(f"Error during office file conversion: {e}", style="bold red", expand=False))
    # 必要に応じて exit() するか、変換失敗ファイルのログを残す


# Officeファイルから画像抽出処理 (.docx, .xlsx, .pptx から画像をサルベージ)
try:
    mo.extract_images(settings['working_dir'])
    console.print(Panel("Officeファイルからの画像抽出が完了しました。", style="bold green", expand=False))
except Exception as e:
    console.print(Panel(f"Error during office image extraction: {e}", style="bold red", expand=False))


# Officeファイルから画像抽出処理 (.docx, .xlsx, .pptx から画像をサルベージ)
try:
    mo.extract_images(settings['backup_dir'])
    console.print(Panel("Officeファイルからの画像抽出が完了しました。", style="bold green", expand=False))
except Exception as e:
    console.print(Panel(f"Error during office image extraction: {e}", style="bold red", expand=False))


# 不要なファイル・フォルダの削除
try:
    mf.remove_unnecessary_data(
        settings['backup_dir'],
        set(settings['unwanted_files']),
        set(settings['image_ext'])
    )
    message = f"Successfully cleaned up files and folders in '{settings['backup_dir']}'."
    console.print(Panel(message, style="bold green", expand=False))
except Exception as e:
    console.print(Panel(f"Error during cleanup: {e}", style="bold red", expand=False))


# サブデータセットの統合と患者フォルダの統合
try:
    source_dirs = [os.path.join(settings['backup_dir'], p) for p in settings['patient_root']]
    target_dir = settings['backup_dir']
    mf.move_patient_folders_to_target(source_dirs, target_dir)
    console.print(Panel("患者フォルダの統合が完了しました。", style="bold green", expand=False))
except Exception as e:
    console.print(Panel(f"Error during patient folder integration: {e}", style="bold red", expand=False))

# 各患者フォルダ内の画像ファイルを直下に移動（フラット化）
try:
    mf.organize_images_in_patient_folders(target_dir, settings['image_ext'])
    console.print(Panel("患者フォルダ内の画像整理が完了しました。", style="bold green", expand=False))
except Exception as e:
    console.print(Panel(f"Error during patient folder image organization: {e}", style="bold red", expand=False))


# 不要なファイル・フォルダの削除
try:
    mf.remove_unnecessary_data(
        settings['backup_dir'],
        set(settings['unwanted_files']),
        set(settings['image_ext'])
    )
    message = f"Successfully cleaned up files and folders in '{settings['backup_dir']}'."
    console.print(Panel(message, style="bold green", expand=False))
except Exception as e:
    console.print(Panel(f"Error during cleanup: {e}", style="bold red", expand=False))