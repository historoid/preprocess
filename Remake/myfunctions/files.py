import os
# import stat
import yaml
import uuid
import shutil
import unicodedata
import pandas as pd

from rich import box
from rich.table import Table
from rich.panel import Panel
from rich.console import Console
from typing import Union, Dict, Set, List, Tuple
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn


# Richコンソールの初期化
console = Console()


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


def get_file_size_str(file_path):
    """ファイルサイズを人間が読みやすい形式で返す"""
    size_bytes = os.path.getsize(file_path)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0 or unit == 'TB':
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0


def copy_directory_with_metadata(src: str, dst: str, overwrite: bool = False) -> None:
    """
    Copies all contents of source directory to destination, preserving metadata.
    """
    assert isinstance(src, str) and isinstance(dst, str), "Source/destination must be strings."
    assert isinstance(overwrite, bool), "'overwrite' parameter must be a boolean."

    if not os.path.exists(src):
        raise FileNotFoundError(f"Source directory '{src}' does not exist.")

    if os.path.exists(dst):
        if overwrite:
            shutil.rmtree(dst)
            print(f"Existing destination '{dst}' removed for overwriting.")
        else:
            raise FileExistsError(f"Destination '{dst}' exists. Use 'overwrite=True'.")

    os.makedirs(dst, exist_ok=True)

    for root, dirs, files in os.walk(src):
        relative_path = os.path.relpath(root, src)
        dst_root = os.path.join(dst, relative_path)

        for dir_name in dirs:
            os.makedirs(os.path.join(dst_root, dir_name), exist_ok=True)

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn()
        ) as progress:
            task = progress.add_task("[green]Copying a file...", total=len(files))

            for file_name in files:
                src_file = os.path.join(root, file_name)
                dst_file = os.path.join(dst_root, file_name)
                file_size = get_file_size_str(src_file)
                progress.update(
                    task,
                    description=f"[green]Copy: [bold white]{file_name} [yellow]({file_size})"
                )
                shutil.copy2(src_file, dst_file)
                progress.update(task, advance=1)
    print(f"Directory '{src}' copied to '{dst}' with overwrite={overwrite}.")


def remove_unwanted_files(root_dir: str, unwanted_files: Set[str], allowed_extensions: Set[str] = None) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    指定された root_dir 以下の全階層を再帰的に探索し、
    ファイル名が unwanted_files に含まれるか、または allowed_extensions が指定されていてファイルの拡張子が含まれていない場合に
    該当ファイルを削除します。

    Args:
        root_dir (str): 探索対象のルートディレクトリ。
        unwanted_files (Set[str]): 削除対象とするファイル名の集合（例：{".DS_Store", "Thumbs.db"}）。
        allowed_extensions (Set[str], optional): 残すべきファイルの拡張子集合。これに含まれないファイルは削除対象となります。

    Returns:
        Tuple[List[str], List[Tuple[str, str]]]:
            - removed_files: 削除に成功したファイルのパスリスト。
            - error_files: 削除時に発生したエラーの (file_path, error_message) のリスト。
    """
    assert isinstance(root_dir, str), "root_dir must be a string."
    assert isinstance(unwanted_files, set), "unwanted_files must be a set."
    
    removed_files: List[str] = []
    error_files: List[Tuple[str, str]] = []
    
    for root, _, files in os.walk(root_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            ext = os.path.splitext(file_name)[1].lower()
            # 削除対象：ファイル名が unwanted_files にある OR
            # allowed_extensions が指定されていて、その拡張子が含まれていない場合
            if file_name in unwanted_files or (allowed_extensions is not None and ext not in allowed_extensions):
                try:
                    os.remove(file_path)
                    removed_files.append(file_path)
                except Exception as e:
                    error_files.append((file_path, str(e)))
    return removed_files, error_files



def is_empty_dir(dir_path: str) -> bool:
    """指定ディレクトリ内に隠しファイル以外のエントリがなければ True を返す。"""
    for entry in os.scandir(dir_path):
        if entry.name in ('.', '..'):
            continue
        if entry.is_symlink():
            try:
                if not os.path.exists(entry.path):
                    continue
            except OSError:
                continue
        return False
    return True


def remove_empty_directories(root_dir: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    指定された root_dir 以下の空ディレクトリを再帰的に削除します。

    Returns:
        Tuple[List[str], List[Tuple[str, str]]]: (removed_dirs, error_dirs)
    """
    assert isinstance(root_dir, str), "root_dir must be a string."

    removed_dirs: List[str] = []
    error_dirs: List[Tuple[str, str]] = []

    with console.status("[bold blue]Searching empty dir...") as status:
        iteration = 1
        while True:
            found_empty_dirs: List[str] = []
            for root, dirs, _ in os.walk(root_dir, topdown=False):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    if is_empty_dir(dir_path):
                        found_empty_dirs.append(dir_path)
            if not found_empty_dirs:
                status.update("[bold green]No more empty directories found.")
                break
            status.update(f"[bold blue]空のディレクトリを検索中... (iteration {iteration})")
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
            ) as progress:
                task = progress.add_task("[yellow]Deleting an empty directory...", total=len(found_empty_dirs))
                for dir_path in found_empty_dirs:
                    dir_name = os.path.basename(dir_path)
                    progress.update(task, description=f"[yellow]Deleting: [bold white]{dir_name}")
                    try:
                        os.rmdir(dir_path)
                        removed_dirs.append(dir_path)
                    except Exception as e:
                        error_dirs.append((dir_path, str(e)))
                    progress.update(task, advance=1)
            iteration += 1

    if removed_dirs:
        table = Table(title="Deleted Directories", box=box.ROUNDED)
        table.add_column("No.", style="cyan")
        table.add_column("Directory path", style="green")
        for i, dir_path in enumerate(removed_dirs, 1):
            table.add_row(str(i), dir_path)
        console.print(Panel(table, title="[bold green]Complete Deletion", subtitle=f"合計: {len(removed_dirs)}ディレクトリ"))
    else:
        console.print("[yellow]There were NO directories to delete")

    if error_dirs:
        error_table = Table(title="Directory with Errors", box=box.ROUNDED)
        error_table.add_column("No.", style="cyan")
        error_table.add_column("Directory path", style="red")
        error_table.add_column("Error Message", style="yellow")
        for i, (dir_path, error) in enumerate(error_dirs, 1):
            error_table.add_row(str(i), dir_path, error)
        console.print(Panel(error_table, title="[bold red]Error", subtitle=f"Total: {len(error_dirs)} directories"))
    return removed_dirs, error_dirs


def remove_small_files(root_dir: str, allowed_extensions: Set[str], size_threshold_kb: int = 300) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    指定されたディレクトリ内で、allowed_extensions に含まれる拡張子を持つファイルのうち、
    サイズが size_threshold_kb 以下のファイルを削除します。

    Returns:
        Tuple[List[str], List[Tuple[str, str]]]: (removed_files, error_files)
    """
    removed_files = []
    error_files = []
    size_threshold_bytes = size_threshold_kb * 1024

    for root, _, files in os.walk(root_dir):
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext in allowed_extensions:
                file_path = os.path.join(root, file_name)
                try:
                    file_size = os.path.getsize(file_path)
                    if file_size <= size_threshold_bytes:
                        os.remove(file_path)
                        removed_files.append(file_path)
                except Exception as e:
                    error_files.append((file_path, str(e)))
    return removed_files, error_files


def remove_unnecessary_data(root_dir: str, unwanted_file_names: Set[str], allowed_extensions: Set[str], size_threshold_kb: int = 100) -> None:
    """
    不要なファイル（不要なファイル名や、allowed_extensions に含まれない拡張子のファイル）、
    小さすぎるファイル、および空のディレクトリを削除します。

    Args:
        root_dir (str): 処理対象のルートディレクトリ。
        unwanted_file_names (Set[str]): 削除対象とするファイル名の集合。
        allowed_extensions (Set[str]): 残すべきファイルの拡張子集合。これに含まれないファイルは削除対象。
        size_threshold_kb (int): 小さすぎるファイルとみなすサイズの閾値（キロバイト）。
    """
    console.print(
        Panel(
            f"[bold cyan]Begin the Cleaning: [white]{root_dir}",
            title="[bold]Cleaning Process",
            subtitle="Deletion of unnecessary files or directories",
        )
    )
    # STEP 1: 不要なファイルの削除（指定ファイル名＋allowed_extensions に含まれないファイル）
    console.print("[bold blue]STEP 1: Deleting unnecessary files...")
    removed_files, error_files = remove_unwanted_files(root_dir, unwanted_file_names, allowed_extensions)
    
    # STEP 2: 小さすぎるファイルの削除（allowed_extensions の対象ファイルのみ）
    console.print(f"[bold blue]STEP 2: Deleting files smaller than {size_threshold_kb}KB...")
    removed_small_files, error_small_files = remove_small_files(root_dir, allowed_extensions, size_threshold_kb)
    
    # STEP 3: 空のディレクトリの削除
    console.print("[bold blue]STEP 3: Deleting empty directories...")
    removed_dirs, error_dirs = remove_empty_directories(root_dir)
    
    # エラー出力
    all_error_files = error_files + error_small_files
    _print_errors(all_error_files, error_dirs, console)
    
    console.print(
        Panel(
            "[bold green]CLEAN UP DONE",
            title="[bold]COMPLETE!!",
            subtitle=f"Target Directory: {root_dir}",
        )
    )


def _print_errors(error_files: List[Tuple[str, str]], error_dirs: List[Tuple[str, str]], console: Console) -> None:
    """エラー内容をコンソールに出力するヘルパー関数"""
    if error_files:
        console.print("[bold red]Files deletion errors:")
        for file_path, error in error_files:
            console.print(f"  [red]- {file_path}: {error}")
    if error_dirs:
        console.print("[bold red]Directory deletion errors:")
        for dir_path, error in error_dirs:
            console.print(f"  [red]- {dir_path}: {error}")
    if not error_files and not error_dirs:
        console.print("[bold green]No errors occurred during the cleanup process.")


def normalize_and_remove_spaces(name: str) -> str:
    """
    Normalizes a file/directory name:
      - Converts full-width to half-width.
      - Removes spaces (both half-width and full-width).

    Args:
        name: The original file/directory name.

    Returns:
        The normalized name.
    """
    normalized_name = unicodedata.normalize("NFKC", name)  # 全角を半角に
    normalized_name = normalized_name.replace(" ", "").replace("　", "").replace("・","")
    return normalized_name


def generate_unique_filename(root: str, new_name: str) -> str:
    """
    Generates a unique filename by appending a counter if necessary.
    Only appends counter if there's name conflict

    Args:
        root: The directory where the file/directory will be renamed.
        new_name: The desired new name (without path).
    Returns:
        A unique filename.
    """

    base, ext = os.path.splitext(new_name)
    counter = 1
    unique_name = new_name
    while os.path.exists(os.path.join(root, unique_name)):
        unique_name = f"{base}({counter}){ext}"
        counter += 1
    return unique_name


def rename_and_normalize(root_dir: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Recursively renames files/directories under root_dir, normalizing names.

    Args:
        root_dir: The root directory to process.

    Returns:
      Tuple: (renamed_entries, error_entries)
          renamed_entries: List of (old_path, new_path) tuples.
          error_entries: List of (path, error_message) tuples.
    """
    renamed_entries: List[Tuple[str, str]] = []
    error_entries: List[Tuple[str, str]] = []

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        total_files = sum(len(files) for _, _, files in os.walk(root_dir))
        total_dirs = sum(len(dirs) for _, dirs, _ in os.walk(root_dir))
        total_items = total_files + total_dirs
        task = progress.add_task("[cyan]Renaming...", total=total_items)

        for root, dirs, files in os.walk(root_dir, topdown=False):
            # Rename files first
            for file_name in files:
                old_path = os.path.join(root, file_name)
                normalized_name = normalize_and_remove_spaces(file_name)

                # 変更がある場合のみ処理
                if normalized_name != file_name:
                    new_name = generate_unique_filename(root, normalized_name) # 重複回避
                    new_path = os.path.join(root, new_name)
                    try:
                        shutil.move(old_path, new_path)
                        renamed_entries.append((old_path, new_path))
                        # コンソールにリネーム情報を表示
                        console.print(f"[green]Renamed file:[/green] {old_path} -> {new_path}")
                        progress.update(task, advance=1, description=f"[green]Renamed File[/]: {new_name}")

                    except Exception as e:
                        error_entries.append((old_path, str(e)))
                        progress.update(task, advance=1, description=f"[red]Error File[/]:{file_name}")
                else:
                    progress.update(task, advance=1)

            # Then rename directories
            for dir_name in dirs:
                old_path = os.path.join(root, dir_name)
                normalized_name = normalize_and_remove_spaces(dir_name)

                # 変更がある場合のみ
                if normalized_name != dir_name:
                    new_name = generate_unique_filename(root, normalized_name) # 重複回避
                    new_path = os.path.join(root, new_name)

                    try:
                        shutil.move(old_path, new_path)
                        renamed_entries.append((old_path, new_path))
                        # コンソールにリネーム情報を表示
                        console.print(f"[green]Renamed dir:[/green] {old_path} -> {new_path}")
                        progress.update(task, advance=1, description=f"[green]Renamed Dir[/]: {new_name}")

                    except Exception as e:
                        error_entries.append((old_path, str(e)))
                        progress.update(task, advance=1, description=f"[red]Error Dir[/]: {dir_name}")
                else:
                    progress.update(task, advance=1)

    return renamed_entries, error_entries


def organize_standalone_files(target_dir: str, allowed_extensions: Set[str] = None) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Organizes standalone files in the target_dir.  Creates a directory
    with the same name as each file (without extension) and moves the file
    into that directory.  Handles name collisions by appending a counter.

    Args:
        target_dir: The directory to process.
        allowed_extensions:  Optional set of extensions to process. If None,
            process all files.

    Returns:
        Tuple: (moved_files, error_files)
            moved_files: List of paths to moved files.
            error_files: List of (file_path, error_message) tuples.

    """
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"Target directory '{target_dir}' not found.")

    moved_files: List[str] = []
    error_files: List[Tuple[str, str]] = []

    for item_name in os.listdir(target_dir):
        item_path = os.path.join(target_dir, item_name)

        if not os.path.isfile(item_path):
            continue

        base_name, ext = os.path.splitext(item_name)

        if allowed_extensions and ext.lower() not in allowed_extensions:
            continue

        new_dir_name = base_name
        new_dir_path = os.path.join(target_dir, new_dir_name)

        counter = 1
        while os.path.exists(new_dir_path):
            new_dir_name = f"{base_name}_{counter}"
            new_dir_path = os.path.join(target_dir, new_dir_name)
            counter += 1

        try:
            os.makedirs(new_dir_path)
            console.print(f"Created directory: {new_dir_path}") # 確認用

            new_file_path = os.path.join(new_dir_path, item_name)
            shutil.move(item_path, new_file_path)
            moved_files.append(new_file_path)
            console.print(f"Moved file: {item_path} -> {new_file_path}") # 確認用

        except Exception as e:
            error_files.append((item_path, str(e)))

    return moved_files, error_files



def organize_patient_data(root_dir: str, patient_root_paths: List[str], allowed_extensions: Set[str] = None) -> None:
    """
    Organizes standalone files within patient root directories.

    Args:
        root_dir:  The main root directory (e.g., "mydata/01_copy").
        patient_root_paths: List of relative paths to patient root directories
                           (from settings.yaml).
        allowed_extensions: Optional set of file extensions to process.
    """

    console.print(Panel("[bold blue]Organizing standalone files...[/]"))
    all_moved_files = []
    all_error_files = []

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:

        task = progress.add_task("[cyan]Processing...", total=len(patient_root_paths))
        for patient_root_rel_path in patient_root_paths:
            patient_root_abs_path = os.path.join(root_dir, patient_root_rel_path)
            # 存在しない、またはディレクトリでない場合はスキップ
            if not os.path.exists(patient_root_abs_path) or not os.path.isdir(patient_root_abs_path):
                console.print(f"[yellow]Skipping (not a directory or not found): {patient_root_rel_path}[/]")
                progress.update(task, advance=1)  # Progress は進める
                continue

            progress.update(task, description=f"[cyan]Processing[/]: {patient_root_rel_path}")
            try:
                moved_files, error_files = organize_standalone_files(patient_root_abs_path, allowed_extensions)
                all_moved_files.extend(moved_files)
                all_error_files.extend(error_files)
            except Exception as e:
                console.print(f"[red]Error processing {patient_root_rel_path}: {e}[/]")
                # 全体としてのエラーリストに追加など、必要に応じて処理
            progress.update(task, advance=1)
            
    if all_moved_files:
        console.print(Panel(f"[green]Moved {len(all_moved_files)} files.[/]", title="[bold]Files Moved"))

    if all_error_files:
        console.print(Panel("[red]Errors occurred during file organization.[/]", title="[bold red]Errors"))
        for file_path, error_msg in all_error_files:
            console.print(f"[red]- {file_path}:[/] {error_msg}")
    elif not all_moved_files: # 移動対象のファイルもエラーもなかった場合
        console.print(Panel("[yellow]No standalone files found to organize.[/]"))



def move_patient_folders_to_target(root_dirs: List[str], target_dir: str) -> None:
    """
    複数の収集元ディレクトリ（root_dirs）内の患者フォルダを、単一の target_dir に移動します。
    
    Parameters:
        root_dirs (List[str]): 患者フォルダが格納されている複数のディレクトリ
        target_dir (str): 患者フォルダを統合する先のディレクトリ

    Raises:
        FileNotFoundError: いずれかのディレクトリが存在しない場合
        ValueError: 引数の型が不正な場合
    """
    # 入力検証
    if not isinstance(root_dirs, list):
        raise ValueError("root_dirs must be a list of paths.")
    if not isinstance(target_dir, str):
        raise ValueError("target_dir must be a string.")
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"Target directory '{target_dir}' does not exist.")

    for root_dir in root_dirs:
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Root directory '{root_dir}' does not exist.")

    total_folders = sum(
        1 for root_dir in root_dirs for item in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, item))
    )

    console.print(Panel(f"統合対象の患者フォルダ数: {total_folders}", style="bold cyan"))

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]患者フォルダを統合中...", total=total_folders)
        for root_dir in root_dirs:
            for item in os.listdir(root_dir):
                item_path = os.path.join(root_dir, item)
                if not os.path.isdir(item_path):
                    continue

                target_path = os.path.join(target_dir, item)
                # 名前の重複を避ける
                unique_target_path = target_path
                counter = 1
                while os.path.exists(unique_target_path):
                    unique_target_path = f"{target_path}_{counter}"
                    counter += 1

                try:
                    shutil.move(item_path, unique_target_path)
                    console.print(f"[green]移動成功:[/] '{item_path}' → '{unique_target_path}'")
                except Exception as e:
                    console.print(f"[red]移動失敗:[/] '{item_path}' の移動中にエラー: {e}")
                progress.advance(task)
    console.print(Panel("全ての患者フォルダの統合が完了しました。", style="bold green"))



def organize_images_in_patient_folders(parent_folder: str, allowed_extensions: List[str]) -> None:
    """
    患者フォルダ内のすべての画像ファイルを、各患者フォルダ直下に移動してフラットな構造に整えます。
    
    Parameters:
        parent_folder (str): 患者フォルダが格納されているルートディレクトリ
        allowed_extensions (List[str]): 処理対象とする画像ファイルの拡張子リスト（例：[".jpg", ".png", ...]）

    Raises:
        FileNotFoundError: parent_folder が存在しない場合
        ValueError: 引数の型が不正な場合
    """
    if not isinstance(parent_folder, str):
        raise ValueError("parent_folder must be a string.")
    if not isinstance(allowed_extensions, list):
        raise ValueError("allowed_extensions must be a list.")
    if not os.path.exists(parent_folder):
        raise FileNotFoundError(f"The folder '{parent_folder}' does not exist.")

    patient_folders = [
        os.path.join(parent_folder, folder)
        for folder in os.listdir(parent_folder)
        if os.path.isdir(os.path.join(parent_folder, folder))
    ]
    total_folders = len(patient_folders)
    console.print(Panel(f"処理対象の患者フォルダ数: {total_folders}", style="bold cyan"))

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]患者フォルダ内を整理中...", total=total_folders)
        for patient_folder_path in patient_folders:
            console.print(f"[blue]処理中:[/] {patient_folder_path}")
            # 再帰的にファイルを探索して、ルート以外の位置にある画像ファイルを移動
            for root, _, files in os.walk(patient_folder_path):
                # ルート（患者フォルダ直下）は除外
                if os.path.abspath(root) == os.path.abspath(patient_folder_path):
                    continue
                for file_name in files:
                    ext = os.path.splitext(file_name)[1].lower()
                    if ext not in allowed_extensions:
                        continue

                    src_path = os.path.join(root, file_name)
                    dst_path = os.path.join(patient_folder_path, file_name)
                    unique_dst_path = dst_path
                    counter = 1
                    while os.path.exists(unique_dst_path):
                        base, ext = os.path.splitext(dst_path)
                        unique_dst_path = f"{base}_{counter}{ext}"
                        counter += 1

                    try:
                        shutil.move(src_path, unique_dst_path)
                        console.print(f"[green]移動:[/] '{src_path}' → '{unique_dst_path}'")
                    except Exception as e:
                        console.print(f"[red]エラー:[/] '{src_path}' の移動中にエラー: {e}")
            progress.advance(task)
    console.print(Panel("患者フォルダ内の整理が完了しました。", style="bold green"))




def create_dataset_metadata(folder_path: str, allowed_extensions: list) -> pd.DataFrame:
    """
    Creates a dataset metadata DataFrame by recording file information for files with specified extensions.

    Parameters:
        folder_path (str): Path to the folder containing files.
        allowed_extensions (list): List of allowed file extensions (e.g., [".jpg", ".png"]).

    Returns:
        pd.DataFrame: A DataFrame containing metadata of the specified files.

    Raises:
        FileNotFoundError: If the folder_path does not exist.
        ValueError: If allowed_extensions is not a list or folder_path is not a string.
    """
    # Input validation
    if not isinstance(folder_path, str):
        raise ValueError("folder_path must be a string.")
    if not isinstance(allowed_extensions, list):
        raise ValueError("allowed_extensions must be a list.")
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")

    # Initialize a list to hold file metadata
    metadata_list = []

    # Walk through the folder and collect metadata for allowed file types
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            # Check file extension
            ext = os.path.splitext(file_name)[1].lower()
            if ext not in allowed_extensions:
                continue

            # Gather file metadata
            file_path = os.path.join(root, file_name)
            stat_info = os.stat(file_path)
            relative_path = os.path.relpath(file_path, start=os.getcwd())
            parent_folder = os.path.basename(os.path.dirname(file_path))

            metadata_list.append({
                "original_filename": file_name,
                "original_created_date": pd.Timestamp(stat_info.st_ctime, unit='s'),
                "original_updated_date": pd.Timestamp(stat_info.st_mtime, unit='s'),
                "original_filesize": round(stat_info.st_size / 1024, 2),  # Size in KB
                "original_filepath": relative_path,
                "original_parent": parent_folder
            })

    # Convert metadata to DataFrame
    df_metadata = pd.DataFrame(metadata_list)
    return df_metadata



def anonymize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Anonymizes a dataset by generating anonymized folder names and filenames.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing at least 'original_parent' and 'original_created_date'.

    Returns:
        pd.DataFrame: DataFrame with added 'anonymized_parent' and 'anonymized_filename' columns.
    """
    # Validate required columns
    if "original_parent" not in df.columns or "original_created_date" not in df.columns:
        raise ValueError("The DataFrame must contain 'original_parent' and 'original_created_date' columns.")

    # Ensure original_created_date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["original_created_date"]):
        df["original_created_date"] = pd.to_datetime(df["original_created_date"])

    # Generate anonymized_parent values
    unique_parents = df["original_parent"].unique()
    anonymized_map = {parent: str(uuid.uuid4())[:12] for parent in unique_parents}
    df["anonymized_parent"] = df["original_parent"].map(anonymized_map)

    # Generate anonymized_filename values
    def assign_filenames(group):
        group = group.sort_values("original_created_date").reset_index(drop=True)
        group["anonymized_filename"] = group.index + 1
        group["anonymized_filename"] = group["anonymized_filename"].apply(lambda x: f"{x:03}")  # Format as '001', '002', etc.
        return group

    df = df.groupby("anonymized_parent").apply(assign_filenames).reset_index(drop=True)

    return df

















































