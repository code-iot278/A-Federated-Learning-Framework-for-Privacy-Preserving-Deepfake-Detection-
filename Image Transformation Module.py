import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# === Input and Output Folders ===
main_input_folder = ''
main_output_folder = ''

# === Define transformations ===
def color_transform(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + 10) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.2, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * 0.9, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def downsample(img): return cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
def upsample(img): return cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
def sharpen(img):
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    return cv2.filter2D(img, -1, kernel)
def gaussian_blur(img): return cv2.GaussianBlur(img, (5,5), 1.0)
def denoise(img): return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

transformations = {
    'Original': lambda x: x,
    'Color': color_transform,
    'Downsample': downsample,
    'Upsample': upsample,
    'Sharpen': sharpen,
    'Blur': gaussian_blur,
    'Denoise': denoise
}

# === Gather image paths ===
image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob(os.path.join(main_input_folder, '**', ext), recursive=True))

print(f"📂 Found {len(image_paths)} image(s).")

# === Process and display ===
for path in image_paths:
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        print(f"⚠️ Could not read: {path}")
        continue

    rel_path = os.path.relpath(path, main_input_folder)
    rel_dir = os.path.dirname(rel_path)
    filename = os.path.basename(path)

    transformed_images = {}

    # Apply transformations and save
    for name, func in transformations.items():
        try:
            out_img = func(img_bgr)
            if out_img.shape[:2] != img_bgr.shape[:2]:
                out_img = cv2.resize(out_img, (img_bgr.shape[1], img_bgr.shape[0]))

            # Save image
            save_folder = os.path.join(main_output_folder, name, rel_dir)
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, filename)
            cv2.imwrite(save_path, out_img)

            transformed_images[name] = out_img

        except Exception as e:
            print(f"❌ Error transforming {filename} with {name}: {e}")

    # === Display all transformed images ===
    num_transforms = len(transformed_images)
    fig, axs = plt.subplots(1, num_transforms, figsize=(5*num_transforms, 5))

    if num_transforms == 1:
        axs = [axs]

    for ax, (name, img) in zip(axs, transformed_images.items()):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(name)
        ax.axis('off')

    plt.suptitle(f"Transformations for: {filename}", fontsize=16)
    plt.tight_layout()
    plt.show()