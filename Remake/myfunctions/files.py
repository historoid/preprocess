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


def remove_unwanted_files(root_dir: str, unwanted_files: Set[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Removes specific unwanted files (like .DS_Store, Thumbs.db) from the specified root directory.

    Args:
        root_dir (str): The root directory to process.
        unwanted_files (Set[str]): A set of *filenames* (not extensions) to remove.  e.g., {".DS_Store", "Thumbs.db"}

    Returns:
        Tuple[List[str], List[Tuple[str, str]]]: (removed_files, error_files)
            removed_files: List of paths to removed files.
            error_files: List of (file_path, error_message) tuples.

    """
    assert isinstance(root_dir, str), "root_dir must be a string."
    assert isinstance(unwanted_files, set), "unwanted_files must be a set."

    removed_files: List[str] = []
    error_files: List[Tuple[str, str]] = []

    for root, _, files in os.walk(root_dir):
        for file_name in files:
            if file_name in unwanted_files:  # ファイル名で判定
                file_path = os.path.join(root, file_name)
                try:
                    os.remove(file_path)
                    removed_files.append(file_path)
                except Exception as e:
                    error_files.append((file_path, str(e)))

    # (Progress, Table 表示は省略.  必要に応じて追加してください)

    return removed_files, error_files



def is_empty_dir(dir_path: str) -> bool:
    """Checks if a directory is empty, considering hidden files and directories."""
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
    Recursively removes empty directories from the specified root directory.

    Parameters:
        root_dir (str): The root directory to process.
    Returns:
        Tuple[List[str], List[Tuple[str, str]]]: (removed_dirs, error_dirs)
            removed_dirs: List of paths to removed directories
            error_dirs: List of (dir_path, error_message) tuples

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
                task = progress.add_task(
                    "[yellow]Deleting an empty directory...", total=len(found_empty_dirs)
                )

                for dir_path in found_empty_dirs:
                    dir_name = os.path.basename(dir_path)
                    progress.update(
                        task, description=f"[yellow]Deleting: [bold white]{dir_name}"
                    )

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

        console.print(
            Panel(
                table,
                title="[bold green]Complete Deletion",
                subtitle=f"合計: {len(removed_dirs)}ディレクトリ",
            )
        )
    else:
        console.print("[yellow]There were NO directories to delete")

    if error_dirs:
        error_table = Table(title="Directory with Errors", box=box.ROUNDED)
        error_table.add_column("No.", style="cyan")
        error_table.add_column("Directory path", style="red")
        error_table.add_column("Error Message", style="yellow")

        for i, (dir_path, error) in enumerate(error_dirs, 1):
            error_table.add_row(str(i), dir_path, error)

        console.print(
            Panel(
                error_table, title="[bold red]Error", subtitle=f"Total: {len(error_dirs)}directories"
            )
        )
    return removed_dirs, error_dirs # 戻り値を返す


def remove_unnecessary_data(root_dir: str, unwanted_file_names: Set[str]) -> None:
    """
    Removes unwanted files and empty directories.

    Args:
        root_dir (str): The root directory to process.
        unwanted_file_names (Set[str]): Set of filenames to remove.
    """
    console.print(
        Panel(
            f"[bold cyan]Begin the Cleaning: [white]{root_dir}",
            title="[bold]Cleaning Process",
            subtitle="Deletion of unnecessary files or directories",
        )
    )
    # Step 1: Remove unwanted files
    console.print("[bold blue]STEP 1: Deleting unnecessary files...")
    removed_files, error_files = remove_unwanted_files(root_dir, unwanted_file_names)
    # Step 2: Remove empty directories
    console.print("[bold blue]STEP 2: Deleting empty directories...")
    removed_dirs, error_dirs = remove_empty_directories(root_dir)
    # エラー出力
    _print_errors(error_files, error_dirs, console)
    console.print(
        Panel(
            "[bold green]CLEAN UP DONE",
            title="[bold]COMPLETE!!",
            subtitle=f"Target Directory: {root_dir}",
        )
    )


def _print_errors(error_files, error_dirs, console):
    """Helper function to print errors"""
    if error_files:
        console.print("[bold red]Errors occurred while deleting files:[/]")
        for file_path, error_msg in error_files:
            console.print(f"  [red]- {file_path}:[/] {error_msg}")

    if error_dirs:
        console.print("[bold red]Errors occurred while deleting directories:[/]")
        for dir_path, error_msg in error_dirs:
            console.print(f"  [red]- {dir_path}:[/] {error_msg}")

    if not error_files and not error_dirs:
        console.print("[bold green]No errors occurred during the cleanup process.[/]")


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
    normalized_name = normalized_name.replace(" ", "").replace("　", "")  # スペース除去
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



def move_patient_folders_to_target(root_dirs: list, target_dir: str) -> None:
    """
    Moves all patient folders from multiple root directories to a single target directory.
    
    Parameters:
        root_dirs (list): A list of root directories containing patient folders.
        target_dir (str): The directory where all patient folders will be moved.
    
    Raises:
        FileNotFoundError: If any of the root directories or the target directory does not exist.
        ValueError: If root_dirs is not a list or target_dir is not a string.
    """
    # Input validation
    if not isinstance(root_dirs, list):
        raise ValueError("root_dirs must be a list of paths.")
    if not isinstance(target_dir, str):
        raise ValueError("target_dir must be a string.")
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"Target directory '{target_dir}' does not exist.")

    for root_dir in root_dirs:
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Root directory '{root_dir}' does not exist.")

    # Move each patient folder
    for root_dir in root_dirs:
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)

            # Skip non-directory items
            if not os.path.isdir(item_path):
                continue

            # Define the target path for the patient folder
            target_path = os.path.join(target_dir, item)

            # Handle name conflicts by appending a unique number
            unique_target_path = target_path
            counter = 1
            while os.path.exists(unique_target_path):
                unique_target_path = f"{target_path}_{counter}"
                counter += 1

            # Move the folder
            shutil.move(item_path, unique_target_path)
            print(f"Moved '{item_path}' to '{unique_target_path}'")

    print("All patient folders have been successfully moved.")



def organize_images_in_patient_folders(parent_folder: str, allowed_extensions: list) -> None:
    """
    Moves image files from subdirectories to the patient folder root.

    Parameters:
        parent_folder (str): Path to the folder containing patient folders.
        allowed_extensions (list): List of allowed file extensions (e.g., [".jpg", ".png"]).

    Raises:
        FileNotFoundError: If the parent folder does not exist.
        ValueError: If allowed_extensions is not a list or parent_folder is not a string.
    """
    # Input validation
    if not isinstance(parent_folder, str):
        raise ValueError("parent_folder must be a string.")
    if not isinstance(allowed_extensions, list):
        raise ValueError("allowed_extensions must be a list.")
    if not os.path.exists(parent_folder):
        raise FileNotFoundError(f"The folder '{parent_folder}' does not exist.")

    # Process each patient folder
    for patient_folder in os.listdir(parent_folder):
        patient_folder_path = os.path.join(parent_folder, patient_folder)

        # Skip non-directory items
        if not os.path.isdir(patient_folder_path):
            continue

        print(f"Processing patient folder: {patient_folder_path}")

        # Traverse all files within the patient folder
        for root, _, files in os.walk(patient_folder_path):
            for file_name in files:
                # Check file extension
                ext = os.path.splitext(file_name)[1].lower()
                if ext not in allowed_extensions:
                    continue

                # Define source and destination paths
                src_path = os.path.join(root, file_name)
                dst_path = os.path.join(patient_folder_path, file_name)

                # Handle name conflicts
                unique_dst_path = dst_path
                counter = 1
                while os.path.exists(unique_dst_path):
                    unique_dst_path = f"{os.path.splitext(dst_path)[0]}_{counter}{ext}"
                    counter += 1

                # Move the file
                shutil.move(src_path, unique_dst_path)
                print(f"Moved '{src_path}' to '{unique_dst_path}'")

    print("Image organization process completed.")



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

















































