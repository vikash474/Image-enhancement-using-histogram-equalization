import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy

# Load the original and processed images
original_image = cv2.imread('processedimage/1.png', cv2.IMREAD_GRAYSCALE)
processed_image = cv2.imread('processedimage/13.png', cv2.IMREAD_GRAYSCALE)

# Ensure both images have the same dimensions
processed_image = cv2.resize(processed_image, (original_image.shape[1], original_image.shape[0]))

# Calculate PSNR
def calculate_psnr(original_image, processed_image):
    mse = np.mean((original_image - processed_image) ** 2)
    max_pixel_value = 255.0
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

# Calculate SSIM
def calculate_ssim(original_image, processed_image):
    ssim_value, _ = ssim(original_image, processed_image, full=True)
    return ssim_value

# Calculate Entropy
def calculate_entropy(image):
    return entropy(image.flatten())

# Compute PSNR, SSIM, and Entropy for both images
psnr_original = calculate_psnr(original_image, original_image)
psnr_processed = calculate_psnr(original_image, processed_image)

ssim_original = calculate_ssim(original_image, original_image)
ssim_processed = calculate_ssim(original_image, processed_image)

entropy_original = calculate_entropy(original_image)
entropy_processed = calculate_entropy(processed_image)

# Print the PSNR, SSIM, and Entropy values
print("Original Image:")
print(f"PSNR Value: {psnr_original:.2f}")
print(f"SSIM Value: {ssim_original:.4f}")
print(f"Entropy Value: {entropy_original:.2f}")
print("\nProcessed Image:")
print(f"PSNR Value: {psnr_processed:.2f}")
print(f"SSIM Value: {ssim_processed:.4f}")
print(f"Entropy Value: {entropy_processed:.2f}")
