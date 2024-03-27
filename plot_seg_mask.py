import cv2
import numpy as np
from PIL import Image
import os
import glob

BASE_DIRECTORY = "/home/chris-lai/yolov8/converted_data/YOLOv8_finetune.v1i.yolov8-car-label/train/"
IMAGE_DIR = os.path.join(BASE_DIRECTORY, "images")
LABEL_DIR = os.path.join(BASE_DIRECTORY, "labels")
OUTPUT_DIR = "./temp_output/"


def parse_seg_label_file(seg_label_path, image_shape):
    """
    Parses a segmentation label file.

    Parameters:
    - seg_label_path: Path to the segmentation label file.
    - image_shape: Tuple of (height, width) of the corresponding image.

    Returns:
    - masks: A list of mask arrays.
    - class_ids: A list of class IDs associated with each mask.
    """
    masks = []
    class_ids = []

    with open(seg_label_path, 'r') as file:
        for line in file:
            parts = line.split()
            class_id = int(parts[0])
            # Assuming the rest of the line is normalized mask coordinates
            coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
            # Un-normalize coordinates
            coords[:, 0] *= image_shape[1]  # Width
            coords[:, 1] *= image_shape[0]  # Height
            # Create a blank mask
            mask = np.zeros(image_shape, dtype=np.uint8)
            # Draw the polygon on the mask
            coords = np.array([coords], dtype=np.int32)  # cv2.fillPoly expects a 3D array
            cv2.fillPoly(mask, coords, 1)
            masks.append(mask)
            class_ids.append(class_id)

    return masks, class_ids

def save_all_masks_on_one_image(raw_image, masks, save_dir, save_filename="all_masks_overlay"):
    raw_image_array = np.array(raw_image, dtype=np.float32)
    overlay_image = raw_image_array.copy()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for mask in masks:
        color = np.random.randint(0, 255, size=3).tolist()
        color = np.array([*color, 153.0])
        alpha = 0.6

        mask = mask.astype(np.float32)
        h, w = mask.shape

        colored_mask = np.zeros((h, w, 4), dtype=np.float32)
        colored_mask[..., :3] = color[:3]
        colored_mask[..., 3] = mask * color[3]

        alpha_blend = colored_mask[..., 3:] / 255.0
        overlay_image = (1 - alpha_blend) * overlay_image + alpha_blend * colored_mask[..., :3]

    overlay_image = np.clip(overlay_image, 0, 255).astype(np.uint8)
    overlay_pil = Image.fromarray(overlay_image)
    save_path = os.path.join(save_dir, save_filename + ".png")
    overlay_pil.save(save_path)
    print(f"Saved: {save_path}")

def process_images(label_filenames):
    for label_filename in label_filenames:
        image_filename = label_filename.replace(".txt", ".jpg")  # Assuming image files are PNG
        image_path = os.path.join(IMAGE_DIR, image_filename)
        label_path = os.path.join(LABEL_DIR, label_filename)

        if os.path.exists(image_path) and os.path.exists(label_path):
            raw_image = Image.open(image_path).convert("RGB")
            image_shape = np.array(raw_image).shape[:2]

            masks, _ = parse_seg_label_file(label_path, image_shape)

            save_filename = os.path.splitext(label_filename)[0]
            save_all_masks_on_one_image(raw_image, masks, OUTPUT_DIR, save_filename=save_filename)
        else:
            print(f"Skipping {label_filename}, corresponding image or label file not found. in {image_path}")

label_filenames = [
    'frame_001128_PNG.rf.768150f279cc5e122034fecc78ffde0d.txt', 'frame_001082_PNG.rf.955ec61e4c8563bb4559a83a301a2c4e.txt', 'frame_001125_PNG.rf.b8c3f956a68763c3778858405b9bc8b2.txt', 'frame_001095_PNG.rf.168da6fb0ef03d9379d05ae04de488da.txt', 'frame_001117_PNG.rf.9f368ac407460a0520049a1c9eb15ebb.txt', 'frame_001111_PNG.rf.fd5e24548b1693c84e34f901a7991c87.txt', 'frame_001120_PNG.rf.f0a47e03a8085628337410b9ffd62209.txt', 'frame_001127_PNG.rf.4d2fd6e5794b7153136ab0a5273b144d.txt', 'frame_001087_PNG.rf.8ba5437558becc6201c86211bd8e00aa.txt', 'frame_001085_PNG.rf.4d103b49c8a4622873f939d6652cba29.txt', 'frame_000790_PNG.rf.4632f3516ac1f00887feb00d2fcd2334.txt', 'frame_002371_PNG.rf.4f56eb3e73cc6b3cff37ab894e752f9e.txt', 'frame_001104_PNG.rf.c470257b0e04eb943e122be24c5c4266.txt', 'frame_001084_PNG.rf.b733849ddf0a20ba4fde9ebd23f60d8b.txt', 'frame_001077_PNG.rf.91e64b7782d62d52a299c3cb35716fbb.txt', 'frame_001100_PNG.rf.a3485f33493b670a1c95d5b1c2d04b17.txt', 'frame_001136_PNG.rf.c7eaf0d08e72109cb006eb7489c87904.txt', 'frame_001134_PNG.rf.ab7a95b88d9ef2aabd978e654d32ce11.txt', 'frame_001106_PNG.rf.940d6375659294c9346f6f9a42ba6238.txt', 'frame_001130_PNG.rf.887592318c6775e66a55b0b03e86db2a.txt', 'frame_001099_PNG.rf.97169c64c51776f3e9393bbca9c68a3c.txt', 'frame_001103_PNG.rf.45640b82a7f0a63ec6a17898b07634c4.txt', 'frame_001126_PNG.rf.af211113e7ee7c0f26e54811966127f8.txt', 'frame_000399_PNG.rf.900674300308e25c044bea4352eb3d6a.txt', 'frame_001102_PNG.rf.8d85ddc01ac57395d7971da0919a86e5.txt', 'frame_000794_PNG.rf.5074968e7a60cb087be510ac9f3f4c38.txt', 'frame_000398_PNG.rf.2dae3e61fed50236551bda96afdfd137.txt', 'frame_001122_PNG.rf.513d0ec169d655c5c19a2fbfce0aa899.txt', 'frame_001112_PNG.rf.0574d43494f43b3a680006d10a316aa9.txt', 'frame_000791_PNG.rf.b6a634f8ef3184344e226ee2958085af.txt', 'frame_001129_PNG.rf.dbd1d795445a08abcd9b485431757b25.txt', 'frame_001135_PNG.rf.5dfbef938629bcafe5c2bb4388a7cac2.txt', 'frame_001105_PNG.rf.4a7fc742306fbbef56575b6834bf4ef9.txt', 'frame_001076_PNG.rf.6d6b6a699ab8dba950d40a83e6a2f80a.txt', 'frame_001108_PNG.rf.9c265e88e1090d0f2d36e9611a32b952.txt', 'frame_001115_PNG.rf.c25174fba039ed48a11e1d5b3f2a700b.txt', 'frame_001123_PNG.rf.401811b458339492dc45666c8aef102e.txt', 'frame_000397_PNG.rf.5c2992f2aa32a337f7a74656740512ab.txt', 'frame_001081_PNG.rf.f6468a21773fd55288cd26e77b524d63.txt', 'frame_001090_PNG.rf.f22f232621ca5d401ecdd62604c8f521.txt'
]

if __name__ == "__main__":
    process_images(label_filenames)