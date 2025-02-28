import os
import stat
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


def remove_unnecessary_data(root_dir: str) -> None:
    """
    Removes unwanted files and empty directories.
    """
    unwanted_file_names = {
        ".DS_Store",
        "Thumbs.db",
        "desktop.ini",
        "ehthumbs.db",
        "ehthumbs_vista.db",
        "$RECYCLE.BIN",
        ".Trash",
        ".localized",
        ".thumbnails"
    } #削除したいファイル名

    console.print(
      Panel(
        f"[bold cyan]Begin the Cleaning: [white]{root_dir}",
        title="[bold]Cleaning Process",
        subtitle="Deletion of unnecessary files or directories",
    )
    )

    # Step 1: Remove unwanted files
    console.print("[bold blue]STEP 1: Deleting unnecessary files...")
    removed_files, error_files = remove_unwanted_files(root_dir, unwanted_file_names) # 変更

    # Step 2: Remove empty directories
    console.print("[bold blue]STEP 2: Deleting empty directories...")
    removed_dirs, error_dirs = remove_empty_directories(root_dir) #変更

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



def remove_spaces_in_names(root_dir: str) -> None:
    """
    Removes spaces (both half-width and full-width) from file and directory names
    within the specified root directory, only if spaces are present. Resolves name
    conflicts by appending a number.

    Parameters:
        root_dir (str): The root directory to process.
    """
    assert isinstance(root_dir, str), "The root directory path must be a string."
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"The specified directory '{root_dir}' does not exist.")
    
    # Walk through the directory structure, bottom-up to handle nested directories
    for root, dirs, files in os.walk(root_dir, topdown=False):
        # Rename files
        for file_name in files:
            if " " in file_name or "　" in file_name:  # Only process if spaces are present
                old_path = os.path.join(root, file_name)
                new_name = file_name.replace(" ", "").replace("　", "")  # Remove spaces
                new_path = os.path.join(root, new_name)
                
                # Resolve conflicts by appending a number
                count = 1
                while os.path.exists(new_path):
                    name, ext = os.path.splitext(new_name)
                    new_path = os.path.join(root, f"{name}({count}){ext}")
                    count += 1
                
                # Rename the file
                if old_path != new_path:
                    os.rename(old_path, new_path)
                    print(f"Renamed file: '{old_path}' -> '{new_path}'")
        
        # Rename directories
        for dir_name in dirs:
            if " " in dir_name or "　" in dir_name:  # Only process if spaces are present
                old_path = os.path.join(root, dir_name)
                new_name = dir_name.replace(" ", "").replace("　", "")  # Remove spaces
                new_path = os.path.join(root, new_name)
                
                # Resolve conflicts by appending a number
                count = 1
                while os.path.exists(new_path):
                    new_path = os.path.join(root, f"{new_name}({count})")
                    count += 1
                
                # Rename the directory
                if old_path != new_path:
                    os.rename(old_path, new_path)
                    print(f"Renamed directory: '{old_path}' -> '{new_path}'")

    print("Space removal and renaming complete.")



def to_half_width(text: str) -> str:
    """
    Converts full-width alphanumeric characters and symbols to half-width.
    
    Parameters:
        text (str): The input string to convert.
    
    Returns:
        str: The converted string with half-width characters.
    """
    return unicodedata.normalize('NFKC', text)

def rename_to_half_width(root_dir: str) -> None:
    """
    Renames all files and directories within the specified root directory,
    converting full-width alphanumeric characters and symbols to half-width.
    Resolves name conflicts by appending a number (e.g., (1), (2)).
    
    Parameters:
        root_dir (str): The root directory to process.
    """
    assert isinstance(root_dir, str), "The root directory path must be a string."
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"The specified directory '{root_dir}' does not exist.")
    
    # Rename files first to avoid downstream path issues
    for root, dirs, files in os.walk(root_dir, topdown=False):  # Bottom-up to handle nested directories
        # Rename files
        for file_name in files:
            old_path = os.path.join(root, file_name)
            new_name = to_half_width(file_name)
            if old_path != os.path.join(root, new_name):  # Only rename if conversion changes the name
                new_path = os.path.join(root, new_name)
                
                # Resolve conflicts
                count = 1
                while os.path.exists(new_path):
                    name, ext = os.path.splitext(new_name)
                    new_path = os.path.join(root, f"{name}({count}){ext}")
                    count += 1
                
                os.rename(old_path, new_path)
                print(f"Renamed file: '{old_path}' -> '{new_path}'")
        
        # Rename directories
        for dir_name in dirs:
            old_path = os.path.join(root, dir_name)
            new_name = to_half_width(dir_name)
            if old_path != os.path.join(root, new_name):  # Only rename if conversion changes the name
                new_path = os.path.join(root, new_name)
                
                # Resolve conflicts
                count = 1
                while os.path.exists(new_path):
                    new_path = os.path.join(root, f"{new_name}({count})")
                    count += 1
                
                os.rename(old_path, new_path)
                print(f"Renamed directory: '{old_path}' -> '{new_path}'")
    
    print("Full-width to half-width renaming complete.")



def organize_office_files(root_folders: list) -> None:
    """
    Organizes standalone office files in the root folders by moving them into
    newly created folders with the same name as the file.

    Parameters:
        root_folders (list): A list of root folder paths to process.

    Supported extensions: .doc, .docx, .xls, .xlsx, .ppt, .pptx
    """
    assert isinstance(root_folders, list), "root_folders must be a list of folder paths."
    supported_extensions = {".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"}

    for root_folder in root_folders:
        if not os.path.exists(root_folder):
            print(f"Skipping non-existent folder: {root_folder}")
            continue

        print(f"Processing folder: {root_folder}")

        # List all files in the root folder
        for item in os.listdir(root_folder):
            item_path = os.path.join(root_folder, item)

            # Skip directories
            if os.path.isdir(item_path):
                continue

            # Check file extension
            _, ext = os.path.splitext(item)
            if ext.lower() not in supported_extensions:
                continue

            # Create a new folder with the same name as the file (excluding extension)
            folder_name = os.path.splitext(item)[0]
            new_folder_path = os.path.join(root_folder, folder_name)

            # Handle name conflicts
            count = 1
            while os.path.exists(new_folder_path):
                new_folder_path = os.path.join(root_folder, f"{folder_name}({count})")
                count += 1

            # Create the new folder
            os.makedirs(new_folder_path)
            print(f"Created folder: {new_folder_path}")

            # Move the file into the newly created folder
            new_file_path = os.path.join(new_folder_path, item)
            shutil.move(item_path, new_file_path)
            print(f"Moved file: '{item_path}' -> '{new_file_path}'")

    print("Office file organization complete.")



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

















































