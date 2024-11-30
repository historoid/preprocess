import cv2
import mahotas
import imagehash

import numpy as np
import pandas as pd
import polars as pl

from PIL import Image
from umap import UMAP
from tqdm import tqdm
from skimage.feature import hog
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops


# 画像の読み込み
def load_image(path, library="opencv"):
    """
    Load an image using OpenCV or Pillow.

    Args:
        path: Path to the image file.
        library: The library to use for loading the image. 
                 Options: "opencv" (default) or "pillow".

    Returns:
        The loaded image in the specified format, or None if loading fails.

    Raises:
        ValueError: If an unsupported library is specified.
    """
    try:
        if library == "opencv":
            image = cv2.imread(path)
            if image is None:
                print(f"Failed to load image using OpenCV: {path}")
            return image
        elif library == "pillow":
            image = Image.open(path)
            return image
        else:
            raise ValueError(f"Unsupported library specified: {library}. Use 'opencv' or 'pillow'.")
    except Exception as e:
        print(f"Error loading image {path} with {library}: {e}")
        return None


# 画像をリサイズして長辺を指定サイズに
def resize_image(image, max_length=128):
    """
    Resize an image so that its longest side matches the specified max_length,
    while maintaining the aspect ratio.

    Supports both OpenCV (NumPy array) and Pillow (PIL.Image.Image) formats.

    Args:
        image: The image to resize (OpenCV or Pillow format).
        max_length: The target maximum length for the longest side of the image.

    Returns:
        The resized image in the same format as the input.
    """
    if isinstance(image, np.ndarray):  # OpenCV format
        h, w = image.shape[:2]
        if max(h, w) > max_length:
            scale = max_length / max(h, w)
            new_size = (int(w * scale), int(h * scale))  # (width, height)
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return image
    elif isinstance(image, Image.Image):  # Pillow format
        w, h = image.size  # Pillow uses (width, height)
        if max(h, w) > max_length:
            scale = max_length / max(h, w)
            new_size = (int(w * scale), int(h * scale))  # (width, height)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        return image
    else:
        raise TypeError("Unsupported image format. Expected OpenCV (NumPy) or Pillow (PIL.Image.Image).")


# PCAによる次元削減
def pca_reduction(features, n_components):
    """
    Apply PCA for dimensionality reduction.

    Args:
        features: A 2D numpy array or list-like structure.
        n_components: Number of components to reduce to.

    Returns:
        Reduced features as a numpy array.
    
    Raises:
        ValueError: If features is not a list or numpy array.
        ValueError: If features is not a 2D structure.
    """
    # Check if features is a list or numpy array
    if not isinstance(features, (list, np.ndarray)):
        raise ValueError("features must be a list or numpy array.")
    
    # Convert list to numpy array if necessary
    if isinstance(features, list):
        features = np.array(features)

    # Check if features is 2D
    if features.ndim != 2:
        raise ValueError("features must be a 2D array or list of lists.")
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    
    return reduced_features


# UMAPによる次元削減
def umap_reduction(features, n_components=32, n_neighbors=15, min_dist=0.1, metric='euclidean'):
    """
    Apply UMAP for dimensionality reduction.

    Args:
        features: A 2D numpy array or list-like structure.
        n_components: Number of components to reduce to (default=2).
        n_neighbors: The size of local neighborhood used for manifold approximation (default=15).
        min_dist: The minimum distance between points in the low-dimensional space (default=0.1).
        metric: The metric to measure distance in the input space (default='euclidean').

    Returns:
        Reduced features as a numpy array.
    
    Raises:
        ValueError: If features is not a list or numpy array.
        ValueError: If features is not a 2D structure.
    """
    # Check if features is a list or numpy array
    if not isinstance(features, (list, np.ndarray)):
        raise ValueError("features must be a list or numpy array.")
    
    # Convert list to numpy array if necessary
    if isinstance(features, list):
        features = np.array(features)

    # Check if features is 2D
    if features.ndim != 2:
        raise ValueError("features must be a 2D array or list of lists.")
    
    # Apply UMAP
    umap = UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric)
    reduced_features = umap.fit_transform(features)
    
    return reduced_features


# Noneをカウントする
def count_none_in_column(df, column_name):
    """
    Count the number of None (or NaN) values in a specific column of the DataFrame.

    Args:
        df: The DataFrame (Polars or Pandas).
        column_name: The name of the column to check for None values.

    Returns:
        int: The number of None (or NaN) values in the specified column.
    """
    if isinstance(df, pl.DataFrame):  # Polarsの場合
        none_count = df.select(pl.col(column_name).is_null().sum()).item()
    elif isinstance(df, pd.DataFrame):  # Pandasの場合
        none_count = df[column_name].isnull().sum()
    else:
        raise TypeError("The input DataFrame must be either Polars or Pandas DataFrame.")
    
    return none_count


# ゼロ埋めベクトルをカウントするため
def count_zeros_in_column(df, column_name):
    """
    Count the number of zero-filled vectors in a specific column of the DataFrame.

    Args:
        df: The DataFrame (Polars or Pandas).
        column_name: The name of the column to check for zero-filled vectors.

    Returns:
        int: The number of zero-filled vectors in the specified column.
    """
    zero_count = 0
    
    if isinstance(df, pl.DataFrame):  # Polarsの場合
        zero_count = df.select(
            pl.col(column_name).map_elements(
                lambda x: all(v == 0 for v in x) if isinstance(x, list) else False,
                return_dtype=pl.Boolean
            ).sum()
        ).item()
    
    elif isinstance(df, pd.DataFrame):  # Pandasの場合
        zero_count = sum(
            df[column_name].apply(lambda x: isinstance(x, list) and all(v == 0 for v in x))
        )
    
    else:
        raise TypeError("The input DataFrame must be either Polars or Pandas DataFrame.")
    
    return zero_count


# 次元が揃っているか確認
def check_dimensions(df, column_name):
    """
    Check if all vectors in a specified column have the same dimension based on the most frequent dimension.

    Supports both Polars and Pandas DataFrames.

    Args:
        df: DataFrame (Polars or Pandas).
        column_name: The name of the column to check.

    Returns:
        A dictionary containing the total number of rows, the count of rows with mismatched dimensions,
        and the indices of mismatched rows (for debugging or further inspection).
    """
    # Initialize counts
    total_rows = len(df)
    mismatch_count = 0
    mismatch_indices = []

    # Check for Polars or Pandas
    if isinstance(df, pl.DataFrame):
        # Get the lengths of each vector in the column
        lengths = df.select(
            pl.col(column_name).map_elements(len, return_dtype=pl.Int64).alias("lengths")  # 修正ポイント
        )
        # Find the most common length (mode)
        most_common_length = lengths["lengths"].mode()[0]
        # Filter out rows with mismatched lengths
        mismatches = lengths.filter(pl.col("lengths") != most_common_length)
        mismatch_count = mismatches.height
        mismatch_indices = mismatches.to_dict()["lengths"]
    
    elif isinstance(df, pd.DataFrame):
        # Get the lengths of each vector in the column
        lengths = df[column_name].apply(lambda x: len(x) if isinstance(x, list) else -1)
        # Find the most common length (mode)
        most_common_length = lengths.mode()[0]
        # Identify rows with mismatched lengths
        mismatches = lengths != most_common_length
        mismatch_indices = df.index[mismatches].tolist()
        mismatch_count = len(mismatch_indices)
    
    else:
        raise TypeError("The input DataFrame must be a Polars or Pandas DataFrame.")

    # Prepare result
    result = {
        "total_rows": total_rows,
        "mismatch_count": mismatch_count,
        "mismatch_indices": mismatch_indices,
    }

    # Print the results
    print(f"Total rows in {column_name}: {total_rows}")
    print(f"Number of rows with dimension mismatch in {column_name}: {mismatch_count}")
    return result


# 画像の色ヒストグラムを計算
def calculate_color_histogram(image):
    """
    Calculate the color histogram for a resized image.

    Args:
        image: Resized image as a NumPy array or PIL.Image.Image.

    Returns:
        Flattened histogram as a list, or None if the image is invalid or None.
    """
    if image is None:
        print("Error: The input image is None.")
        return None

    # Convert PIL.Image to NumPy array if necessary
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Check if the image is a valid NumPy array with 3 channels
    if not isinstance(image, np.ndarray):
        print("Error: The input image is not a valid NumPy array.")
        return None
    if image.ndim != 3 or image.shape[2] != 3:
        print("Error: The input image must have 3 channels (BGR or RGB).")
        return None

    # Initialize an empty list to store the histogram
    histogram = []
    try:
        for channel in range(3):  # B, G, R channels
            hist = cv2.calcHist([image], [channel], None, [128], [0, 256]).flatten()
            histogram.extend(hist)
        return histogram
    except Exception as e:
        print(f"Error during histogram calculation: {e}")
        return None


# Zスコア正規化
def normalize_zscore(features):
    """
    Apply Z-score normalization (standardization) to a 1D or 2D array or list-like structure.

    Args:
        features: A 1D or 2D numpy array or list-like structure.

    Returns:
        Normalized features as a numpy array.

    Raises:
        ValueError: If features is not a list or numpy array.
    """
    # Check if features is a list or numpy array
    if not isinstance(features, (list, np.ndarray)):
        raise ValueError("features must be a list or numpy array.")
    
    # Convert list to numpy array if necessary
    if isinstance(features, list):
        features = np.array(features)

    # Handle 1D input by reshaping to 2D
    if features.ndim == 1:
        features = features.reshape(-1, 1)

    # Apply standardization
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    
    # Flatten back to 1D if original input was 1D
    if normalized_features.shape[1] == 1:
        normalized_features = normalized_features.flatten()
    
    return normalized_features



# Perceptual Hash値を計算
def calculate_perceptual_hash(image, hash_size=32):
    """
    Calculate the Perceptual Hash (pHash) for a given image.

    Args:
        image: Pillow Image object (grayscale, resized if needed).
        hash_size: Size of the hash, default is 32 (32x32).

    Returns:
        A string representation of the perceptual hash, or None if an error occurs.
    """
    try:
        # Ensure the image is grayscale and resized appropriately
        if image.mode != "L":
            image = image.convert("L")
        image = image.resize((hash_size, hash_size), Image.Resampling.LANCZOS)

        # Calculate the Perceptual Hash
        hash_value = imagehash.phash(image)
        return str(hash_value)  # Return the hash as a string

    except Exception as e:
        print(f"Error calculating perceptual hash: {e}")
        return None



# ORB特徴量の算出
# 画像によって出力次元が可変
def extract_orb_descriptors(image, n_features=500):
    """
    Extract ORB descriptors from a given image.

    Args:
        image: Loaded image (grayscale) as a NumPy array.
        n_features: Number of features to detect with ORB. Default is 500.

    Returns:
        descriptors: ORB descriptors as a NumPy array (None if no descriptors found).
    """
    try:
        orb = cv2.ORB_create(nfeatures=n_features)
        _, descriptors = orb.detectAndCompute(image, None)
        return descriptors  # None if no keypoints are detected
    except Exception as e:
        print(f"Error extracting ORB descriptors: {e}")
        return None


# コードブックの生成
def create_codebook_orb(image_paths, n_clusters=128, n_features=500, max_length=128):
    """
    Create a codebook (cluster centers) from ORB descriptors of a set of images.

    Args:
        image_paths: List of paths to the input images.
        n_clusters: Number of clusters for the codebook. Default is 128.
        n_features: Number of features to detect with ORB. Default is 500.
        max_length: Maximum size (pixels) for the longest side of the image. Default is 256.

    Returns:
        A NumPy array of cluster centers (codebook).

    Raises:
        ValueError: If no valid descriptors are found in the provided images.
    """
    all_descriptors = []

    for path in image_paths:
        # Load image
        image = load_image(path, library="opencv")
        if image is None:
            print(f"Warning: Could not load image at path: {path}")
            continue

        # Resize image for consistency
        resized_image = resize_image(image, max_length=max_length)

        # Initialize ORB
        orb = cv2.ORB_create(nfeatures=n_features)

        # Extract ORB descriptors
        keypoints, descriptors = orb.detectAndCompute(resized_image, None)
        if descriptors is not None and len(descriptors) > 0:
            all_descriptors.append(descriptors)
        else:
            print(f"Warning: No descriptors found for image at path: {path}")

    # Check if descriptors were collected
    if not all_descriptors:
        raise ValueError("No valid descriptors found in the provided images. "
                         "Ensure that the images contain detectable features.")

    # Combine all descriptors
    all_descriptors = np.vstack(all_descriptors)

    # Fit K-means to generate the codebook
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(all_descriptors)

    return kmeans.cluster_centers_



# ORB特徴量を固定次元化
def compute_histogram_from_descriptors(descriptors, codebook):
    """
    Compute a histogram from ORB descriptors using a given codebook.

    Args:
        descriptors: ORB descriptors as a NumPy array.
        codebook: Precomputed codebook (cluster centers) as a NumPy array.

    Returns:
        A 1D NumPy array representing the histogram.
    """
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(len(codebook))  # Zero vector for images without descriptors

    # Compute distances to codebook centers and assign clusters
    distances = cdist(descriptors, codebook)
    nearest_clusters = np.argmin(distances, axis=1)

    # Generate histogram
    hist, _ = np.histogram(nearest_clusters, bins=np.arange(len(codebook) + 1))
    return hist.astype(np.float32)



# ガンマ補正
def apply_gamma_correction(image, gamma=1.5):
    """
    Apply gamma correction to adjust the brightness of an image.

    Args:
        image: Image as a NumPy array (OpenCV format) or PIL.Image.Image (Pillow format).
        gamma: Gamma value for correction. Values >1 make the image brighter, <1 make it darker.

    Returns:
        Brightness-adjusted image in the same format as the input.
    """
    try:
        # Convert Pillow image to NumPy array
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Ensure image is a valid NumPy array
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a NumPy array or PIL.Image.Image.")

        # Apply gamma correction
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        adjusted_image = cv2.LUT(image, table)

        # Convert back to Pillow if the input was Pillow
        if isinstance(image, Image.Image):
            return Image.fromarray(adjusted_image)

        return adjusted_image
    except Exception as e:
        print(f"Error applying gamma correction: {e}")
        return image  # Return the original image if an error occurs


# 明るさ補正
def enhance_contrast_with_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Enhance the contrast of a grayscale image using CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        image: Image as a NumPy array (OpenCV format) or PIL.Image.Image (Pillow format).
        clip_limit: Threshold for contrast clipping. Default is 2.0.
        tile_grid_size: Size of the grid for histogram equalization. Default is (8, 8).

    Returns:
        Contrast-enhanced image in the same format as the input.
    """
    try:
        # Convert Pillow image to NumPy array
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Ensure the image is in grayscale
        if image.ndim == 3:  # Convert to grayscale if it's a color image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_image = clahe.apply(image)

        # Convert back to Pillow if the input was Pillow
        if isinstance(image, Image.Image):
            return Image.fromarray(enhanced_image)

        return enhanced_image
    except Exception as e:
        print(f"Error enhancing contrast: {e}")
        return image  # Return the original image if an error occurs


# ガンマ補正と明るさ調整
def preprocess_image(image, gamma=1.5, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply gamma correction and CLAHE for preprocessing an image.

    Args:
        image: Image as a NumPy array (OpenCV format) or PIL.Image.Image (Pillow format).
        gamma: Gamma value for brightness adjustment.
        clip_limit: Threshold for contrast clipping in CLAHE.
        tile_grid_size: Size of the grid for histogram equalization in CLAHE.

    Returns:
        Preprocessed image in the same format as the input.
    """
    try:
        # Apply gamma correction
        image = apply_gamma_correction(image, gamma=gamma)

        # Apply contrast enhancement
        image = enhance_contrast_with_clahe(image, clip_limit=clip_limit, tile_grid_size=tile_grid_size)

        return image
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return image  # Return the original image if an error occurs


# エッジ検出
def detect_edges(image, method="canny", threshold1=50, threshold2=150):
    """
    Detect edges in the image using the specified method.

    Args:
        image: Input image as a NumPy array (grayscale or BGR).
        method: Edge detection method. Options are "canny" (default).
        threshold1: First threshold for the hysteresis procedure (for Canny).
        threshold2: Second threshold for the hysteresis procedure (for Canny).

    Returns:
        Edges as a binary NumPy array, or None if detection fails.
    """
    if image is None:
        print("Error: The input image is None.")
        return None

    if image.ndim == 3:  # Convert BGR to grayscale if needed
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if method == "canny":
        try:
            edges = cv2.Canny(image, threshold1, threshold2)
            return edges
        except Exception as e:
            print(f"Error during edge detection: {e}")
            return None
    else:
        print(f"Error: Unsupported edge detection method '{method}'.")
        return None


# エッジのヒストグラムを計算
def calculate_edge_histogram(edges, bins=128, range=(0, 256)):
    """
    Calculate the histogram of edge intensities.

    Args:
        edges: Binary edge image as a NumPy array.
        bins: Number of bins for the histogram.
        range: Intensity range for the histogram.

    Returns:
        Flattened histogram as a list, or None if the input is invalid.
    """
    if edges is None:
        print("Error: The input edges array is None.")
        return None
    
    try:
        # Flatten the array and calculate the histogram
        hist, _ = np.histogram(edges.ravel(), bins=bins, range=range)
        return hist.tolist()
    except Exception as e:
        print(f"Error during edge histogram calculation: {e}")
        return None



# 輝度の平均と分散を計算
def calculate_brightness_stats(image):
    """
    Calculate the mean and variance of brightness for a given image.

    Args:
        image: Input image as a NumPy array (BGR or grayscale).

    Returns:
        A tuple (mean_brightness, variance_brightness), or None if input is invalid.
    """
    if image is None:
        print("Error: The input image is None.")
        return None

    # グレースケール変換
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 輝度の平均と分散を計算
    mean_brightness = np.mean(image)
    variance_brightness = np.var(image)

    return mean_brightness, variance_brightness


# GLCM特徴量
def calculate_glcm_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256):
    """
    Calculate GLCM features for a grayscale image.

    Args:
        image: Grayscale image as a 2D NumPy array.
        distances: List of pixel distances for GLCM calculation.
        angles: List of angles (in radians) for GLCM calculation.
        levels: Number of gray levels for quantization (default: 256).

    Returns:
        A flattened NumPy array of GLCM features.
    """
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale image.")

    # Compute GLCM
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)

    # Extract GLCM properties
    features = []
    for prop in ['contrast', 'homogeneity', 'energy', 'correlation']:
        feature = graycoprops(glcm, prop).flatten()
        features.extend(feature)

    return np.array(features, dtype=np.float32)


# LBP特徴量
def calculate_lbp_histogram(image, radius=1, n_points=8, method="uniform", expected_dim=59):
    """
    Calculate the Local Binary Pattern (LBP) histogram for a grayscale image.

    Args:
        image: Grayscale image as a NumPy array.
        radius: Radius of the LBP pattern.
        n_points: Number of points in the LBP pattern.
        method: Method to compute the LBP ('default', 'ror', 'uniform', 'var').
        expected_dim: Expected dimension of the output histogram (default=59).

    Returns:
        LBP histogram as a 1D NumPy array with fixed dimensions (expected_dim).
    """
    if image is None or image.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale image.")

    # Calculate LBP
    lbp = local_binary_pattern(image, n_points, radius, method)

    # Compute histogram
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))

    # Normalize the histogram
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-7  # Avoid division by zero

    # Pad or truncate the histogram to match the expected dimension
    if len(hist) < expected_dim:
        hist = np.pad(hist, (0, expected_dim - len(hist)), 'constant')
    elif len(hist) > expected_dim:
        hist = hist[:expected_dim]

    return hist


# Gaborフィルタ特徴量
def apply_gabor_filters(image, frequencies, thetas):
    """
    Apply Gabor filters to a grayscale image and extract statistical features.

    Args:
        image: Grayscale image as a NumPy array.
        frequencies: List of frequencies for the Gabor filters.
        thetas: List of orientations (angles in radians) for the Gabor filters.

    Returns:
        A 1D NumPy array containing mean and standard deviation for each filter.
    """
    if image is None or image.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale image.")

    features = []
    for frequency in frequencies:
        for theta in thetas:
            # Create Gabor kernel
            kernel = cv2.getGaborKernel(
                ksize=(31, 31),  # Kernel size
                sigma=4.0,  # Standard deviation of the Gaussian envelope
                theta=theta,  # Orientation
                lambd=1.0 / frequency,  # Wavelength
                gamma=0.5,  # Aspect ratio
                psi=0  # Phase offset
            )
            # Apply Gabor filter
            filtered_image = cv2.filter2D(image, cv2.CV_64F, kernel)
            # Calculate statistical features
            features.append(filtered_image.mean())
            features.append(filtered_image.std())

    return np.array(features, dtype=np.float32)


# HOG特徴量
def calculate_hog_features(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    """
    Calculate HOG features for a given image.

    Args:
        image: Grayscale image as a NumPy array.
        pixels_per_cell: Size of the cell in pixels (default: 8x8).
        cells_per_block: Size of the block in cells (default: 2x2).
        orientations: Number of orientation bins (default: 9).

    Returns:
        HOG features as a 1D NumPy array.
    """
    if image is None or image.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale image.")
    
    # Calculate HOG features
    try:
        features = hog(
            image,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            orientations=orientations,
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True
        )
        return features
    except Exception as e:
        print(f"Error calculating HOG features: {e}")
        return None


# HOG特徴量の次元揃え（ゼロパディングとクリッピング）
def align_feature_dimensions(features, target_length):
    """
    Align feature dimensions by applying zero-padding or clipping.

    Args:
        features: A list or NumPy array of feature vectors.
        target_length: Target length for all feature vectors.

    Returns:
        A 2D NumPy array where all vectors have the target length.
    """
    aligned_features = []
    for feature in features:
        if len(feature) < target_length:
            # ゼロパディング
            padded_feature = np.pad(feature, (0, target_length - len(feature)), mode="constant")
            aligned_features.append(padded_feature)
        elif len(feature) > target_length:
            # クリッピング
            clipped_feature = feature[:target_length]
            aligned_features.append(clipped_feature)
        else:
            # そのまま
            aligned_features.append(feature)
    return np.array(aligned_features)


# Zornike Moments特徴量
def calculate_zernike_moments(image, radius=64, degree=8):
    """
    Calculate Zernike Moments for a grayscale image.

    Args:
        image: Grayscale image as a NumPy array.
        radius: Radius of the circular mask (default=64).
        degree: Maximum degree of Zernike polynomials to calculate (default=8).

    Returns:
        Zernike Moments as a NumPy array or None if calculation fails.
    """
    if image is None or image.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale image.")

    # Resize the image to fit within the circular mask
    try:
        resized_image = cv2.resize(image, (2 * radius, 2 * radius))

        # Apply Otsu's thresholding to create a binary image
        threshold = mahotas.thresholding.otsu(resized_image)
        binary_image = resized_image > threshold

        # Calculate Zernike Moments
        moments = mahotas.features.zernike_moments(binary_image, radius, degree)
        return np.array(moments)
    except Exception as e:
        print(f"Error calculating Zernike Moments: {e}")
        return None









