import cv2
import os
import argparse
from pathlib import Path
from insightface.app import FaceAnalysis
import numpy as np

# python tools/get_faces_img.py image_path.jpg -o output_dir -s 100
def detect_and_save_faces(image_path, output_dir="faces_output", min_size=100):
    """
    Detect faces from an image and save each face.
    
    Args:
        image_path (str): Input image path.
        output_dir (str): Output directory.
        min_size (int): Minimum size (in pixels) of the face image. Faces smaller than this will be ignored.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        return
    
    # Convert to RGB format (insightface requires RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize face detector (using insightface for better performance)
    print("Initializing face detector...")
    face_analyzer = FaceAnalysis(root="pretrained/face_encoder", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(320, 320))
    
    # Detect faces
    print("Detecting faces...")
    face_info = face_analyzer.get(image_rgb)
    
    print(f"Detected {len(face_info)} faces")
    
    # Sort by x-coordinate (from left to right)
    if len(face_info) > 1:
        face_info = sorted(face_info, key=lambda x: (x['bbox'][0] + x['bbox'][2]) / 2)
        print("Faces sorted by position from left to right")
    
    # Get the base name of the input image (without extension)
    base_name = Path(image_path).stem
    
    saved_count = 0
    ignored_count = 0
    
    for i, face in enumerate(face_info):
        # Get face bounding box
        bbox = face['bbox']
        x, y, x2, y2 = bbox
        
        # Ensure coordinates are within image boundaries
        height, width = image_rgb.shape[:2]
        x = max(0, int(x))
        y = max(0, int(y))
        x2 = min(width, int(x2))
        y2 = min(height, int(y2))
        
        w = x2 - x
        h = y2 - y
        
        # Check if the bounding box is valid
        if w <= 0 or h <= 0:
            print(f"Face {i+1}: Invalid bounding box, skipped")
            ignored_count += 1
            continue
        
        # Check if the face size meets the minimum requirement
        if w < min_size or h < min_size:
            print(f"Face {i+1}: Size {w}x{h}, smaller than the minimum size {min_size}x{min_size}, ignored")
            ignored_count += 1
            continue
        
        # Extract face region (convert back from RGB to BGR for saving)
        face_roi_rgb = image_rgb[y:y2, x:x2]
        
        # Check if the extracted region is empty
        if face_roi_rgb.size == 0:
            print(f"Face {i+1}: Extracted face region is empty, skipped")
            ignored_count += 1
            continue
            
        face_roi = cv2.cvtColor(face_roi_rgb, cv2.COLOR_RGB2BGR)
        print(f"Face {i+1}: Size {w}x{h}, meets requirements")
        
        # Save face image
        face_filename = f"{base_name}_face_{saved_count+1}.jpg"
        face_path = os.path.join(output_dir, face_filename)
        
        success = cv2.imwrite(face_path, face_roi)
        if success:
            print(f"Saved face image: {face_path}")
            saved_count += 1
        else:
            print(f"Error: Could not save face image {face_path}")
    
    print(f"Successfully saved {saved_count} face images to the {output_dir} directory")
    if ignored_count > 0:
        print(f"Ignored {ignored_count} faces smaller than the minimum size")
    
    return saved_count

def main():
    parser = argparse.ArgumentParser(description="Detect and save faces from an image")
    parser.add_argument("image_path", help="Input image path")
    parser.add_argument("-o", "--output", default="demo_examples/faces", help="Output directory (default: faces_output)")
    parser.add_argument("-s", "--min-size", type=int, default=70, help="Minimum size of face image (default: 100)")
    
    args = parser.parse_args()
    
    # Check if the input file exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file {args.image_path} does not exist")
        return
    
    # Detect and save faces
    detect_and_save_faces(args.image_path, args.output, args.min_size)

if __name__ == "__main__":
    main()
