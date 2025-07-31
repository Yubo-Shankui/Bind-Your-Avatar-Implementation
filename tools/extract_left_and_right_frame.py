from __future__ import division
import cv2
from PIL import Image
import os
import numpy as np



def get_left_and_right_frame(input_image_path, video_length=None, sample_size=None, fps=None, validation_video_mask=None):
    if isinstance(input_image_path, str):
        image = cv2.imread(input_image_path)
        if image is None:
            raise ValueError(f" can not read any image from the input_image_path. Please check again."
            )
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        left_frame = frame[:, :width//2]
        right_frame = frame[:, width//2:]

    else:
        raise ValueError(f" input_image_path is not of type str. Please check again."
            )
    return left_frame, right_frame


if __name__ == "__main__": 
    image_path           = "assets/inpaintingframe/003.png"
    save_path           = "assets/inpaintingframe/003_left.png"
    left_frame, right_frame = get_left_and_right_frame(image_path)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    base_name = os.path.splitext(save_path)[0]
    extension = os.path.splitext(save_path)[1]
    left_save_path = f"{base_name}_left{extension}"
    right_save_path = f"{base_name}_right{extension}"

    left_image = Image.fromarray(left_frame)
    right_image = Image.fromarray(right_frame)

    left_image.save(left_save_path)
    right_image.save(right_save_path)

    print(f"Left frame successfully saved to {left_save_path}")
    print(f"Right frame successfully saved to {right_save_path}")
