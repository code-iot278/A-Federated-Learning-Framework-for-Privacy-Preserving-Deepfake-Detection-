import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from skimage.feature import hog
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import tensorflow as tf

# === Input & Output Paths ===
main_input_folder = ''
output_csv = ''

# === Load Image Paths ===
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob(os.path.join(main_input_folder, '**', ext), recursive=True))

print(f"\U0001F4C2 Total images found: {len(image_paths)}")

# === Load VGG16 Model ===
vgg_model = VGG16(weights='imagenet', include_top=True)
heatmap_model = Model([vgg_model.inputs], [vgg_model.get_layer("block5_conv3").output, vgg_model.output])

# === Helper: Grad-CAM attention map ===
def get_attention_map(img_resized, class_index):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = heatmap_model(tf.convert_to_tensor(np.expand_dims(img_resized, axis=0)))
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / tf.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    return heatmap

# === Feature Extraction Function ===
def extract_features(image_path):
    try:
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

        # HOG
        hog_feat, hog_img = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2), visualize=True, feature_vector=True)
        hog_mean = np.mean(hog_feat)
        hog_std = np.std(hog_feat)

        # Canny Edge
        canny_edges = cv2.Canny(gray, 100, 200)
        edge_count = np.sum(canny_edges > 0)

        # VGG16 Prediction
        img_input = preprocess_input(np.expand_dims(img_resized.astype(np.float32), axis=0))
        conv_out, predictions = heatmap_model.predict(img_input)
        pred_class = int(np.argmax(predictions[0]))
        pred_conf = float(predictions[0][pred_class])

        # Attention Map
        heatmap = get_attention_map(preprocess_input(img_resized.astype(np.float32)), pred_class)
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)

        # Display
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].imshow(img_rgb)
        axs[0].set_title("Original")
        axs[0].axis('off')

        axs[1].imshow(hog_img, cmap='gray')
        axs[1].set_title("HOG")
        axs[1].axis('off')

        axs[2].imshow(canny_edges, cmap='gray')
        axs[2].set_title("Canny Edges")
        axs[2].axis('off')

        axs[3].imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
        axs[3].set_title("VGG16 + Attention")
        axs[3].axis('off')

        plt.suptitle(os.path.basename(image_path), fontsize=14)
        plt.tight_layout()
        plt.show()

        label = os.path.basename(os.path.dirname(image_path))

        return {
            'filename': os.path.basename(image_path),
            'path': image_path,
            'label': label,
            'hog_feature_mean': hog_mean,
            'hog_feature_std': hog_std,
            'canny_edge_pixel_count': edge_count,
            'vgg16_class_index': pred_class,
            'vgg16_class_confidence': pred_conf
        }

    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return None

# === Run on All Images ===
features = []
for img_path in image_paths:
    print(f"🔍 Processing: {img_path}")
    feats = extract_features(img_path)
    if feats:
        features.append(feats)

# === Save to CSV ===
df = pd.DataFrame(features)
df.to_csv(output_csv, index=False)
print(f"\n✅ All features saved to: {output_csv}")
print(df.head())