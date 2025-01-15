import os
import subprocess
from typing import List
import shutil
import zipfile
from docx import Document
from datetime import datetime



LIBREOFFICE_PATH = '/Applications/LibreOffice.app/Contents/MacOS/soffice'



def get_unique_file_path(file_path: str) -> str:
    """
    Ensures the file path is unique by appending a number if needed.

    Parameters:
        file_path (str): The initial file path.

    Returns:
        str: A unique file path.
    """
    base, ext = os.path.splitext(file_path)
    count = 1
    while os.path.exists(file_path):
        file_path = f"{base}({count}){ext}"
        count += 1
    return file_path

def convert_to_new_format(old_path: str, new_ext: str, libreoffice_path: str = "libreoffice") -> str:
    """
    Converts a file to a new format using LibreOffice.

    Parameters:
        old_path (str): Path to the original file.
        new_ext (str): New file extension (e.g., '.docx').
        libreoffice_path (str): Path to the LibreOffice executable.

    Returns:
        str: Path to the new file, or None if conversion failed.
    """
    # Generate the expected new file path
    new_path = os.path.splitext(old_path)[0] + new_ext
    new_path = get_unique_file_path(new_path)  # Ensure no name conflicts

    # Run LibreOffice conversion
    command = [
        libreoffice_path, '--headless', '--convert-to', new_ext[1:], '--outdir',
        os.path.dirname(old_path), old_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    # Check for errors during conversion
    if result.returncode != 0:
        print(f"Error converting {old_path}: {result.stderr}")
        return None

    # Check if the new file was created successfully
    if not os.path.exists(new_path) or os.path.getsize(new_path) == 0:
        print(f"Conversion failed or resulted in an empty file for {old_path}")
        return None

    return new_path

def convert_office_files(folder_path: str, libreoffice_path: str = LIBREOFFICE_PATH) -> None:
    """
    Converts .doc, .xls, .ppt files in the specified folder (and its subdirectories)
    to .docx, .xlsx, .pptx, removes the original files, and handles name conflicts.

    Parameters:
        folder_path (str): The folder containing the files to convert.
        libreoffice_path (str): Path to the LibreOffice executable.
    """
    assert isinstance(folder_path, str), "folder_path must be a string."
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The specified folder '{folder_path}' does not exist.")

    # Supported file extensions and their new formats
    converters = {
        ".doc": ".docx",
        ".xls": ".xlsx",
        ".ppt": ".pptx"
    }

    failed_files = []  # Keep track of files that failed to convert

    for root, _, files in os.walk(folder_path):  # Use os.walk to explore all subdirectories
        for file_name in files:
            file_path = os.path.join(root, file_name)

            # Check if the file has a supported extension
            _, ext = os.path.splitext(file_name)
            if ext.lower() not in converters:
                continue

            new_ext = converters[ext.lower()]
            print(f"Converting '{file_path}' to '{new_ext}'...")
            new_file_path = convert_to_new_format(file_path, new_ext, libreoffice_path)

            if new_file_path:
                os.remove(file_path)  # Delete the original file
                print(f"Successfully converted and removed: '{file_path}'")
            else:
                print(f"Failed to convert: '{file_path}'")
                failed_files.append(file_path)

    # Log failed files
    if failed_files:
        print("\nThe following files failed to convert:")
        for failed_file in failed_files:
            print(failed_file)



def extract_images_from_docx(docx_path: str, extract_dir: str) -> None:
    """Extract images from .docx files."""
    doc = Document(docx_path)
    image_count = 0
    for rel in doc.part.rels.values():
        if "image" in rel.target_ref:
            image = rel.target_part.blob
            image_format = rel.target_ref.split('.')[-1]
            image_filename = f"image{image_count}.{image_format}"
            image_path = os.path.join(extract_dir, image_filename)
            with open(image_path, 'wb') as image_file:
                image_file.write(image)
            image_count += 1

def extract_images_from_xlsx(xlsx_path: str, extract_dir: str) -> None:
    """Extract images from .xlsx files."""
    with zipfile.ZipFile(xlsx_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.startswith('xl/media/'):
                extracted_file_path = zip_ref.extract(file, extract_dir)
                image_filename = os.path.basename(extracted_file_path)
                new_image_path = os.path.join(extract_dir, image_filename)
                shutil.move(extracted_file_path, new_image_path)

def extract_images_from_pptx(pptx_path: str, extract_dir: str) -> None:
    """Extract images from .pptx files."""
    with zipfile.ZipFile(pptx_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.startswith('ppt/media/'):
                extracted_file_path = zip_ref.extract(file, extract_dir)
                image_filename = os.path.basename(extracted_file_path)
                new_image_path = os.path.join(extract_dir, image_filename)
                shutil.move(extracted_file_path, new_image_path)

def extract_images(folder_path: str) -> None:
    """
    Extracts images from .docx, .xlsx, and .pptx files in the specified folder and its subdirectories,
    and deletes the original files after extraction.

    Parameters:
        folder_path (str): The root folder to process.
    """
    assert isinstance(folder_path, str), "folder_path must be a string."
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The specified folder '{folder_path}' does not exist.")

    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            ext = os.path.splitext(file_name)[1].lower()

            if ext not in {".docx", ".xlsx", ".pptx"}:
                continue

            # Create a folder for extracted images
            extract_folder_name = f"extracted_{os.path.splitext(file_name)[0]}"
            extract_folder_path = os.path.join(root, extract_folder_name)
            os.makedirs(extract_folder_path, exist_ok=True)

            # Process the file and extract images
            try:
                print(f"Processing {file_path}...")
                if ext == ".docx":
                    extract_images_from_docx(file_path, extract_folder_path)
                elif ext == ".xlsx":
                    extract_images_from_xlsx(file_path, extract_folder_path)
                elif ext == ".pptx":
                    extract_images_from_pptx(file_path, extract_folder_path)

                # Delete the original file after successful extraction
                os.remove(file_path)
                print(f"Original file deleted: {file_path}")
                print(f"Images extracted to: {extract_folder_path}")
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

    print("Image extraction process completed.")







































































