import os
import cv2
from glob import glob
from google.colab.patches import cv2_imshow

# === Set paths ===
main_folder = ''
output_root = ''
os.makedirs(output_root, exist_ok=True)

# === Get all video paths from nested folders ===
video_extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv')
video_paths = []

for ext in video_extensions:
    video_paths.extend(glob(os.path.join(main_folder, '**', ext), recursive=True))

print(f"🎞️ Found {len(video_paths)} video(s)")

# === Process each video ===
for video_path in video_paths:
    print(f"🔹 Playing: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Could not open: {video_path}")
        continue

    # Extract subfolder name (Real/Fake) and video filename
    relative_path = os.path.relpath(video_path, main_folder)
    parts = relative_path.split(os.sep)
    category_folder = parts[0]  # e.g., Real or Fake
    video_name = os.path.splitext(parts[-1])[0]  # video file name without extension

    # Output folder for frames
    output_folder = os.path.join(output_root, category_folder, video_name)
    os.makedirs(output_folder, exist_ok=True)

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_num > 100:  # limit to 100 frames
            break

        # Display frame in Colab
        cv2_imshow(frame)

        # Save frame
        frame_filename = os.path.join(output_folder, f"frame_{frame_num:03d}.jpg")
        cv2.imwrite(frame_filename, frame)

        frame_num += 1

        # Optional: Pause briefly between frames
        key = cv2.waitKey(100)
        if key == ord('q'):
            break

    cap.release()
    print(f"✅ Saved {frame_num} frames to: {output_folder}")

cv2.destroyAllWindows()
print("✅ All videos processed.")