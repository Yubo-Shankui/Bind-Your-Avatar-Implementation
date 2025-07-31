import sys
import os
from PIL import Image


def crop_image_half(input_path):
    img = Image.open(input_path)
    w, h = img.size
    mid = w // 2
    left_img = img.crop((0, 0, mid, h))
    right_img = img.crop((mid, 0, w, h))
    base, ext = os.path.splitext(input_path)
    left_path = f"{base}_0{ext}"
    right_path = f"{base}_1{ext}"
    left_img.save(left_path)
    right_img.save(right_path)
    print(f"保存: {left_path}, {right_path}")


if __name__ == "__main__":
    input_path = "tests/input/00013_38_43_0_52.png"
    crop_image_half(input_path)
