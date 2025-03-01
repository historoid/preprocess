import os
import shutil
import zipfile
import subprocess

from typing import List
from docx import Document
from rich.panel import Panel
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

console = Console()
LIBREOFFICE_PATH = '/Applications/LibreOffice.app/Contents/MacOS/soffice'


def get_unique_file_path(file_path: str) -> str:
    """
    指定されたパスが既に存在する場合、番号を付与して一意のパスを生成します。
    """
    base, ext = os.path.splitext(file_path)
    unique_path = file_path
    count = 1
    while os.path.exists(unique_path):
        unique_path = f"{base}({count}){ext}"
        count += 1
    return unique_path

    
def convert_to_new_format(old_path: str, new_ext: str, libreoffice_path: str = LIBREOFFICE_PATH) -> str:
    """
    LibreOffice を用いて古いフォーマットのファイルを新しいフォーマットに変換します。
    Parameters:
        old_path (str): 変換前のファイルパス。
        new_ext (str): 変換後の拡張子（例：'.docx'）。
        libreoffice_path (str): LibreOffice の実行ファイルパス。
    Returns:
        str: 変換後のファイルパス。変換に失敗した場合は None を返します。
    """
    # 変換後のファイルパスを生成
    new_path = os.path.splitext(old_path)[0] + new_ext
    new_path = get_unique_file_path(new_path)
    try:
        command = [
            libreoffice_path,
            '--headless',
            '--convert-to', new_ext[1:],
            '--outdir', os.path.dirname(old_path),
            old_path
        ]
        result = subprocess.run(command, capture_output=True, text=True)
    except Exception as e:
        console.print(f"[red]例外発生:[/] {old_path} の変換中にエラーが発生しました。エラー内容: {e}")
        return None
    if result.returncode != 0:
        console.print(f"[red]変換エラー:[/] {old_path} の変換に失敗しました。エラーメッセージ: {result.stderr}")
        return None
    # 変換後ファイルの存在確認（サイズが 0 でないかもチェック）
    if not os.path.exists(new_path) or os.path.getsize(new_path) == 0:
        console.print(f"[red]変換結果不正:[/] {old_path} の変換後ファイルが作成されなかった、または空のファイルです。")
        return None
    return new_path

def convert_office_files(folder_path: str, libreoffice_path: str = LIBREOFFICE_PATH) -> None:
    """
    指定したフォルダ（およびそのサブディレクトリ）内の .doc, .xls, .ppt ファイルを、
    LibreOffice を使用して .docx, .xlsx, .pptx に変換し、変換後は元のファイルを削除します。
    Parameters:
        folder_path (str): 変換対象となるフォルダパス。
        libreoffice_path (str): LibreOffice の実行ファイルパス。
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"[red]指定されたフォルダ '{folder_path}' は存在しません。[/]")
    # 変換対象の拡張子と新しい拡張子の対応定義
    converters = {
        ".doc": ".docx",
        ".xls": ".xlsx",
        ".ppt": ".pptx"
    }
    failed_files = []  # 変換に失敗したファイルを記録
    converted_count = 0
    # 変換対象となるファイルのパス一覧を収集
    all_files = []
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            _, ext = os.path.splitext(file_name)
            if ext.lower() in converters:
                all_files.append(os.path.join(root, file_name))
    # Rich の Progress を用いて進捗表示
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]オフィスファイルを変換中...", total=len(all_files))
        for file_path in all_files:
            _, ext = os.path.splitext(file_path)
            new_ext = converters[ext.lower()]
            progress.update(task, description=f"[cyan]変換中:[/] {file_path}")
            new_file_path = convert_to_new_format(file_path, new_ext, libreoffice_path)
            if new_file_path:
                try:
                    os.remove(file_path)
                    console.print(f"[green]変換成功:[/] {file_path} → {new_file_path}")
                    converted_count += 1
                except Exception as e:
                    console.print(f"[red]元ファイル削除エラー:[/] {file_path} の削除に失敗しました。エラー内容: {e}")
                    failed_files.append(file_path)
            else:
                console.print(f"[red]変換失敗:[/] {file_path}")
                failed_files.append(file_path)
            progress.advance(task)
    if failed_files:
        console.print(Panel(
            "[red]以下のファイルの変換に失敗しました:\n" + "\n".join(failed_files),
            title="変換失敗",
            style="red"
        ))
    else:
        console.print(Panel(
            f"[green]全てのオフィスファイルの変換が成功しました。（合計: {converted_count}件）",
            title="変換成功",
            style="green"
        ))


def extract_images_from_docx(docx_path: str, extract_dir: str) -> int:
    """
    .docx ファイルから画像を抽出します。
    
    Parameters:
        docx_path (str): 対象の .docx ファイルのパス
        extract_dir (str): 抽出した画像を保存するフォルダ
    
    Returns:
        int: 抽出できた画像の数
    """
    image_count = 0
    try:
        doc = Document(docx_path)
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                image = rel.target_part.blob
                image_format = rel.target_ref.split('.')[-1]
                image_filename = f"image{image_count}.{image_format}"
                image_path = os.path.join(extract_dir, image_filename)
                with open(image_path, 'wb') as image_file:
                    image_file.write(image)
                image_count += 1
    except Exception as e:
        console.print(f"[red]Error extracting images from DOCX '{docx_path}': {e}[/]")
    return image_count


def extract_images_from_xlsx(xlsx_path: str, extract_dir: str) -> int:
    """
    .xlsx ファイルから画像を抽出します。
    
    Parameters:
        xlsx_path (str): 対象の .xlsx ファイルのパス
        extract_dir (str): 抽出した画像を保存するフォルダ
    
    Returns:
        int: 抽出できた画像の数
    """
    image_count = 0
    try:
        with zipfile.ZipFile(xlsx_path, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if file.startswith('xl/media/'):
                    extracted_file_path = zip_ref.extract(file, extract_dir)
                    image_filename = os.path.basename(extracted_file_path)
                    new_image_path = os.path.join(extract_dir, image_filename)
                    shutil.move(extracted_file_path, new_image_path)
                    image_count += 1
    except Exception as e:
        console.print(f"[red]Error extracting images from XLSX '{xlsx_path}': {e}[/]")
    return image_count


def extract_images_from_pptx(pptx_path: str, extract_dir: str) -> int:
    """
    .pptx ファイルから画像を抽出します。
    
    Parameters:
        pptx_path (str): 対象の .pptx ファイルのパス
        extract_dir (str): 抽出した画像を保存するフォルダ
    
    Returns:
        int: 抽出できた画像の数
    """
    image_count = 0
    try:
        with zipfile.ZipFile(pptx_path, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if file.startswith('ppt/media/'):
                    extracted_file_path = zip_ref.extract(file, extract_dir)
                    image_filename = os.path.basename(extracted_file_path)
                    new_image_path = os.path.join(extract_dir, image_filename)
                    shutil.move(extracted_file_path, new_image_path)
                    image_count += 1
    except Exception as e:
        console.print(f"[red]Error extracting images from PPTX '{pptx_path}': {e}[/]")
    return image_count


def extract_images(folder_path: str) -> None:
    """
    指定したフォルダ（およびそのサブディレクトリ）内の .docx, .xlsx, .pptx ファイルから画像を抽出し、
    抽出が成功した場合は元のファイルを削除します。
    
    Parameters:
        folder_path (str): 処理対象のルートフォルダパス
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"[red]The specified folder '{folder_path}' does not exist.[/]")
    
    # 変換対象ファイルのパスを収集
    target_extensions = {".docx", ".xlsx", ".pptx"}
    files_to_process = []
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext in target_extensions:
                files_to_process.append(os.path.join(root, file_name))
    
    failed_files = []
    total_extracted = 0

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]画像抽出処理中...", total=len(files_to_process))
        for file_path in files_to_process:
            ext = os.path.splitext(file_path)[1].lower()
            # 抽出先フォルダの作成（元ファイル名に基づいたフォルダ）
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            extract_folder_name = f"extracted_{base_name}"
            extract_folder_path = os.path.join(os.path.dirname(file_path), extract_folder_name)
            os.makedirs(extract_folder_path, exist_ok=True)
            
            try:
                console.print(f"[blue]Processing:[/] {file_path}")
                extracted = 0
                if ext == ".docx":
                    extracted = extract_images_from_docx(file_path, extract_folder_path)
                elif ext == ".xlsx":
                    extracted = extract_images_from_xlsx(file_path, extract_folder_path)
                elif ext == ".pptx":
                    extracted = extract_images_from_pptx(file_path, extract_folder_path)
                
                if extracted > 0:
                    total_extracted += extracted
                    os.remove(file_path)
                    console.print(f"[green]成功:[/] {file_path} から {extracted} 個の画像を抽出し、元ファイルを削除しました。")
                else:
                    console.print(f"[yellow]警告:[/] {file_path} から画像の抽出が確認できませんでした。")
            except Exception as e:
                console.print(f"[red]失敗:[/] {file_path} の処理中にエラーが発生しました。エラー内容: {e}")
                failed_files.append(file_path)
            progress.advance(task)
    
    if failed_files:
        console.print(Panel(
            "[red]以下のファイルで画像抽出に失敗しました:\n" + "\n".join(failed_files),
            title="抽出失敗",
            style="red"
        ))
    else:
        console.print(Panel(
            f"[green]全てのファイルから画像の抽出が完了しました。（合計 {total_extracted} 個の画像抽出）",
            title="抽出成功",
            style="green"
        ))







































































