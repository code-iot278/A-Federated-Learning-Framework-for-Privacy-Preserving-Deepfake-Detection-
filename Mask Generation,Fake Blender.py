import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

# === Setup MediaPipe Face Mesh ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

# === Set input/output folders ===
main_input_folder = ''
main_output_folder = ''

# === Supported image formats ===
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob(os.path.join(main_input_folder, '**', ext), recursive=True))

print(f"📂 Found {len(image_paths)} images.")

# === Process each image ===
for image_path in image_paths:
    print(f"🖼️ Processing: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Skipping unreadable file: {image_path}")
        continue

    image = cv2.resize(image, (800, 600))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        print("❌ No face detected.")
        continue

    for face_landmarks in results.multi_face_landmarks:
        h, w, _ = image.shape
        points = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
        points_array = np.array(points)
        hull = cv2.convexHull(points_array)

        # === Create all images ===
        original_image = image.copy()

        landmark_image = image.copy()
        mp_drawing.draw_landmarks(
            image=landmark_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1)
        )

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)

        soft_mask = cv2.GaussianBlur(mask, (3, 3), 0)
        segmented_image = cv2.bitwise_and(image, image, mask=soft_mask)

        # === Fake Blender Implementation ===
        target_image_path = np.random.choice([p for p in image_paths if p != image_path])
        target_image = cv2.imread(target_image_path)
        if target_image is None:
            print(f"⚠️ Skipping Fake Blender (target image unreadable): {target_image_path}")
            continue
        target_image = cv2.resize(target_image, (800, 600))

        # I_B = I_(S:M) ⨁ I_(T-M)
        source_face_region = cv2.bitwise_and(image, image, mask=soft_mask)
        inverse_mask = cv2.bitwise_not(soft_mask)
        target_background_region = cv2.bitwise_and(target_image, target_image, mask=inverse_mask)
        blended_image = cv2.add(source_face_region, target_background_region)

        # Post-processing effects
        kernel_sharpening = np.array([[-1, -1, -1],
                                      [-1,  9, -1],
                                      [-1, -1, -1]])
        sharpened_blended = cv2.filter2D(blended_image, -1, kernel_sharpening)

        rgb_shift = blended_image.copy()
        rgb_shift[:, :, 0] = np.clip(rgb_shift[:, :, 0] + 10, 0, 255)
        rgb_shift[:, :, 1] = np.clip(rgb_shift[:, :, 1] - 10, 0, 255)

        dilated_mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        dilated_soft_mask = cv2.GaussianBlur(dilated_mask, (7, 7), 0)
        modified_segmented = cv2.bitwise_and(image, image, mask=dilated_soft_mask)

        # === Save output ===
        relative_path = os.path.relpath(image_path, main_input_folder)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.join(main_output_folder, os.path.dirname(relative_path))
        os.makedirs(output_dir, exist_ok=True)

        # Standard results
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_original.jpg"), original_image)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_landmarks.jpg"), landmark_image)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_binary_mask.jpg"), mask)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_soft_mask.jpg"), soft_mask)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_segmented.jpg"), segmented_image)

        # Fake Blender results
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_fakeblend_original.jpg"), blended_image)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_fakeblend_sharpened.jpg"), sharpened_blended)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_fakeblend_rgbshift.jpg"), rgb_shift)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_fakeblend_maskmod.jpg"), modified_segmented)

        print(f"✅ Saved results to: {output_dir}")

        # === Display Output ===
        fig, axs = plt.subplots(2, 5, figsize=(25, 10))

        axs[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title("Original Image")
        axs[0, 0].axis('off')

        axs[0, 1].imshow(cv2.cvtColor(landmark_image, cv2.COLOR_BGR2RGB))
        axs[0, 1].set_title("Facial Landmarks")
        axs[0, 1].axis('off')

        axs[0, 2].imshow(mask, cmap='gray')
        axs[0, 2].set_title("Binary Mask")
        axs[0, 2].axis('off')

        axs[0, 3].imshow(soft_mask, cmap='gray')
        axs[0, 3].set_title("Soft Mask")
        axs[0, 3].axis('off')

        axs[0, 4].imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
        axs[0, 4].set_title("Segmented Image")
        axs[0, 4].axis('off')

        axs[1, 0].imshow(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB))
        axs[1, 0].set_title("FakeBlender: Blended")
        axs[1, 0].axis('off')

        axs[1, 1].imshow(cv2.cvtColor(sharpened_blended, cv2.COLOR_BGR2RGB))
        axs[1, 1].set_title("FakeBlender: Sharpened")
        axs[1, 1].axis('off')

        axs[1, 2].imshow(cv2.cvtColor(rgb_shift, cv2.COLOR_BGR2RGB))
        axs[1, 2].set_title("FakeBlender: RGB Shift")
        axs[1, 2].axis('off')

        axs[1, 3].imshow(cv2.cvtColor(modified_segmented, cv2.COLOR_BGR2RGB))
        axs[1, 3].set_title("FakeBlender: Mask Mod")
        axs[1, 3].axis('off')

        axs[1, 4].axis('off')  # Empty placeholder

        plt.suptitle(f"Results for {os.path.basename(image_path)}", fontsize=14)
        plt.tight_layout()
        plt.show()