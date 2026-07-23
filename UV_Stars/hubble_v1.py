import os
import gc
import numpy as np
import cv2
import pywt
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import AsinhStretch
from astroquery.mast import Observations
from photutils.detection import DAOStarFinder

plt.ioff()
os.makedirs("rendered", exist_ok=True)

# ---------------------------------------
# Setup Colormap
# ---------------------------------------

# 9-color control points: non-linear distribution (shifted toward mid-tones)
colors_9 = [
    "#7F00FF",  # Pure Black (Background)
    "#5F00FF",  # Deep Violet Void
    "#3F00FF",  # Base Deep Violet (Endpoint 1)
    "#0000FF",
    "#5F5FFF",
    "#AFAFFF",
    "#FFFFFF",  # Base Light Indigo (Endpoint 2)
    "#7FFFFF",  # Soft Glow
    "#00FFFF"   # Star Cores
]
positions_9 = [0.0, 0.05, 0.15, 0.30, 0.50, 0.72, 0.88, 0.95, 1.0]
uvmap_9 = LinearSegmentedColormap.from_list(
    "UV_9_Midtone", list(zip(positions_9, colors_9)), N=512
)

# 12-color control points: logarithmic weighting towards bright regions
colors_12 = [
    "#000000", "#0B0017", "#7F00FF", "#7F11FF", "#7F22FF", "#7F38FF",
    "#7F4FFF", "#7F66FF", "#7F7FFF", "#A3A3FF", "#D9D9FF", "#FFFFFF"
]
positions_12 = [0.0, 0.03, 0.08, 0.15, 0.25, 0.38, 0.53, 0.70, 0.82, 0.90, 0.96, 1.0]
uvmap_12 = LinearSegmentedColormap.from_list(
    "UV_12_Logarithmic", list(zip(positions_12, colors_12)), N=512
)

# 16-color ultra-smooth palette
colors_16 = [
    "#FFFFFF", "#FEFDFF", "#FDFBFF", "#FCF9FF", "#FBF7FF", "#FAF5FF",
    "#F9F3FF", "#F8F1FF", "#F7EEFF", "#F6ECFF", "#F5EAFF", "#F4E8FF",
    "#F3E6FF", "#F2E4FF", "#F1E2FF", "#7F00FF"
]
positions_16 = [
    0.00, 0.02, 0.05, 0.09, 0.14, 0.21, 0.30, 0.41,
    0.53, 0.66, 0.78, 0.86, 0.92, 0.96, 0.98, 1.00
]
uvmap_16 = LinearSegmentedColormap.from_list(
    "UV_16_SCurve", list(zip(positions_16, colors_16)), N=512
)

# Colormap menu -- pick whichever map matches the mood of the dataset:
#   "9"  = high mid-tone structure
#   "12" = focused on bright stars and nebular cores
#   "16" = maximum smooth dynamic range
COLORMAP_CHOICE = "9"
uvmap = {"9": uvmap_9, "12": uvmap_12, "16": uvmap_16}[COLORMAP_CHOICE]

# ---------------------------------------
# Advanced Rendering Pipeline
# ---------------------------------------
def process_hubble_image_advanced(image):
    """
    Advanced NASA-style rendering pipeline optimized for speed and stability.
    """
    # 1. Clean & Cast
    image = image.astype(np.float32)
    image[~np.isfinite(image)] = 0
    
    # Fast Background & Dynamic Range Normalization
    lo, hi = np.percentile(image, (0.2, 99.8))
    if hi > lo:
        image = np.clip((image - lo) / (hi - lo), 0, 1)
    else:
        return image

    # 2. Asinh Stretch
    stretch = AsinhStretch(a=0.08)
    image = stretch(image)

    # 3. Wavelet Detail Enhancement (Fine Filaments)
    try:
        coeffs = pywt.wavedec2(image, "db4", level=3)
        new_coeffs = [coeffs[0]]
        for cH, cV, cD in coeffs[1:]:
            new_coeffs.append((cH * 1.3, cV * 1.3, cD * 1.3))
        image = pywt.waverec2(new_coeffs, "db4")
        image = np.clip(image, 0, 1)
    except Exception:
        pass  # Fallback if image dimensions aren't divisible by wavelet depth

    # 4. Laplacian Pyramid Local Contrast (Reduces noise vs CLAHE)
    try:
        h, w = image.shape
        # Pad slightly if dimensions are odd numbers for cv2.pyrDown/Up
        h_pad, w_pad = h % 2, w % 2
        img_pad = np.pad(image, ((0, h_pad), (0, w_pad)), mode='reflect')
        
        g1 = cv2.pyrDown(img_pad)
        g2 = cv2.pyrDown(g1)
        l0 = img_pad - cv2.pyrUp(g1, dstsize=(img_pad.shape[1], img_pad.shape[0]))
        l1 = g1 - cv2.pyrUp(g2, dstsize=(g1.shape[1], g1.shape[0]))
        
        result = cv2.pyrUp(l1 * 1.2 + g2, dstsize=(img_pad.shape[1], img_pad.shape[0])) + (l0 * 1.4)
        image = np.clip(result[:h, :w], 0, 1)
    except Exception:
        pass

    # 5. Star Detection & Selective Bloom
    try:
        mean, median, std = sigma_clipped_stats(image, sigma=3.0)
        finder = DAOStarFinder(fwhm=3.0, threshold=5.0 * std)
        stars = finder(image - median)
        
        if stars is not None and len(stars) > 0:
            mask = np.zeros_like(image)
            h, w = image.shape
            
            # Draw top 200 brightest stars to keep it fast
            stars.sort("flux")
            top_stars = stars[-200:]
            
            for star in top_stars:
                x, y = int(star["xcentroid"]), int(star["ycentroid"])
                if 0 <= x < w and 0 <= y < h:
                    mask[y, x] = 1.0
            
            # Bloom only point sources
            glow = gaussian_filter(mask * image, sigma=6)
            image = np.clip(image + glow * 0.5, 0, 1)
    except Exception:
        # Fallback to simple highlight threshold if star detection fails
        bright = np.maximum(image - 0.75, 0)
        bloom = gaussian_filter(bright, sigma=6)
        image = np.clip(image + bloom * 0.4, 0, 1)

    return np.clip(image, 0, 1)

# ---------------------------------------
# Batch Processing Settings
# ---------------------------------------
START_PAGE = 3    # Start on page 2 since you already did page 1!
END_PAGE = 4      # How many pages total you want to run
PAGE_SIZE = 12    # 10 observations per batch

for current_page in range(START_PAGE, END_PAGE + 1):
    print(f"\n==========================================")
    print(f"   FETCHING BATCH (PAGE {current_page})")
    print(f"==========================================")

    obs = Observations.query_criteria(
        obs_collection="HST",
        instrument_name=["ACS/WFC", "WFC3/UVIS", "WFC3/IR"],
        pagesize=PAGE_SIZE,
        page=current_page
    )

    if len(obs) == 0:
        print("No more observations returned. Finishing!")
        break

    for idx, single_obs in enumerate(obs):
        obs_num = (current_page - 1) * PAGE_SIZE + idx + 1
        print(f"\n[{obs_num}] Processing observation: {single_obs['obs_id']}")
        
        try:
            products = Observations.get_product_list(single_obs)
            products = Observations.filter_products(
                products,
                productSubGroupDescription=["DRZ", "DRC"],
                extension="fits"
            )
            
            if len(products) == 0:
                print("No DRZ/DRC fits found. Skipping.")
                continue

            download = Observations.download_products(products[:1], mrp_only=True)
            
            for row in download:
                filename = row["Local Path"]
                if not filename.endswith(".fits"):
                    continue

                outfile = os.path.join(
                    "rendered", 
                    os.path.basename(filename).replace(".fits", ".png")
                )
                
                if os.path.exists(outfile):
                    print(f"Already rendered: {outfile}")
                    if os.path.exists(filename):
                        os.remove(filename)
                    continue

                print(f"Downsampling and rendering: {filename}")
                
                image = None
                with fits.open(filename, memmap=True) as h:
                    for ext in h:
                        if ext.data is not None and ext.data.ndim == 2:
                            image = ext.data[::4, ::4].astype(np.float32)
                            break

                if image is None:
                    if os.path.exists(filename):
                        os.remove(filename)
                    continue

                processed_image = process_hubble_image_advanced(image)

                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(processed_image, cmap=uvmap, origin="lower", interpolation="bicubic")
                ax.axis("off")

                plt.savefig(outfile, dpi=200, bbox_inches="tight", pad_inches=0)
                
                plt.close(fig)
                del image, processed_image
                
                if os.path.exists(filename):
                    os.remove(filename)

        except Exception as e:
            print(f"Error processing {single_obs['obs_id']}: {e}")

        gc.collect()

print("\nBatch processing complete!")
