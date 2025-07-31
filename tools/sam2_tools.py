# adapted from https://github.com/PKU-YuanGroup/ConsisID/blob/main/data_preprocess/step4_get_mask.py
import argparse
import gc
import json
import os
import shutil
from pathlib import Path
import cv2

import numpy as np
import supervision as sv
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from tqdm import tqdm
from insightface.app import FaceAnalysis

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def parse_args():
    parser = argparse.ArgumentParser(description="Process video files in a given directory.")
    parser.add_argument("--video_folder", type=str, default="2025_0000.mp4", help="Folder containing video files.")
    parser.add_argument("--output_path", type=str, default="temp/sam2", help="Path to store the output files.")
    parser.add_argument("--sam2_checkpoint_path", type=str, default="pretrained/sam2", help="Path to the SAM2 model checkpoint.")
    parser.add_argument("--model_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml", help="Path to the SAM2 model configuration.")
    return parser.parse_args()

def process_single_video(video_path, output_path, video_predictor, face_analyzer):
    """
    Process a single video file.
    """
    base_name = os.path.basename(video_path.replace(".mp4", ""))
    
    output_video_path = f"{output_path}/{base_name}"
    source_video_frame_dir = f"{output_path}/{base_name}/custom_video_frames"
    save_tracking_mask_results_dir = f"{output_path}/{base_name}"
    save_corresponding_json_dir = f"{output_path}/{base_name}/corresponding_data.json"
    save_control_json_dir = f"{output_path}/{base_name}/control_sam2_frame.json"
    save_bbox_json_dir = f"{output_path}/{base_name}/valid_frame.json"

    if os.path.exists(save_bbox_json_dir) and os.path.exists(save_corresponding_json_dir) and os.path.exists(save_control_json_dir):
        print(f"Skipping processed video: {base_name}")
        return

    try:
        os.makedirs(output_video_path, exist_ok=True)
        os.makedirs(source_video_frame_dir, exist_ok=True)
        os.makedirs(save_tracking_mask_results_dir, exist_ok=True)

        # Read video information
        video_info = sv.VideoInfo.from_video_path(video_path)
        frame_generator = sv.get_video_frames_generator(video_path, stride=1, start=0, end=None)

        # Save video frames
        source_frames = Path(source_video_frame_dir)
        source_frames.mkdir(parents=True, exist_ok=True)

        total_frame_count = video_info.total_frames
        existing_frame_names = [
            p for p in os.listdir(source_video_frame_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        existing_frame_count = len(existing_frame_names)

        if source_frames.exists() and existing_frame_count == total_frame_count:
            print(f"Video frames already exist, skipping extraction: {source_video_frame_dir}")
        else:
            with sv.ImageSink(
                target_dir_path=source_frames,
                overwrite=True,
                image_name_pattern="{:05d}.jpg"
            ) as sink:
                for frame in tqdm(frame_generator, desc="Saving video frames"):
                    sink.save_image(frame)

        # Read the first frame for face detection
        print("Performing face detection...")
        first_frame = cv2.imread(os.path.join(source_video_frame_dir, "00000.jpg"))
        first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        face_info = face_analyzer.get(first_frame_rgb)

        if len(face_info) != 2:
            raise RuntimeError(f"Detected {len(face_info)} faces, exactly 2 faces are required")

        # Sort by x-coordinate (left to right)
        face_info = sorted(face_info, key=lambda x: (x['bbox'][0] + x['bbox'][2]) / 2)

        # Initialize video predictor state
        inference_state = video_predictor.init_state(video_path=source_video_frame_dir, async_loading_frames=True)

        # Prepare data structures
        input_boxes = []
        OBJECT_IDS = []
        OBJECTS = []
        FRAME_IDX = []
        bbox_json_data = {}
        corresponding_json_data = {}
        control_json_data = {}

        # Process each face
        for idx, face in enumerate(face_info, 1):
            bbox = face['bbox']
            input_boxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
            OBJECT_IDS.append(idx)
            OBJECTS.append('face')
            FRAME_IDX.append(0)  # Use the first frame

            if idx not in bbox_json_data:
                bbox_json_data[idx] = {}
                corresponding_json_data[idx] = {}
                control_json_data[idx] = {}

        input_boxes = np.array(input_boxes, dtype=np.float32)

        # Process each detected object
        print("Starting face segmentation...")
        for object_id, track_id, object_name, box, frame_idx in zip(OBJECT_IDS, OBJECT_IDS, OBJECTS, input_boxes, FRAME_IDX):
            print(f"Processing face {object_id}...")
            video_predictor.reset_state(inference_state)

            # Use face keypoints
            face = face_info[object_id-1]
            kps = face['kps']
            points = np.array(kps, dtype=np.float32)
            labels = np.array([1] * len(kps), dtype=np.int32)

            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=object_id,
                box=box,
                points=points,
                labels=labels,
            )

            # Generate segmentation results
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, start_frame_idx=0):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            # Save masks
            temp_save_tracking_mask_results_dir = os.path.join(save_tracking_mask_results_dir, "tracking_mask_results", str(object_id))
            os.makedirs(temp_save_tracking_mask_results_dir, exist_ok=True)

            valid_frame_list = []
            for frame_idx, segments in video_segments.items():
                object_ids = list(segments.keys())
                masks = list(segments.values())
                masks = np.concatenate(masks, axis=0)

                mask_img = torch.zeros(masks.shape[-2], masks.shape[-1])
                mask_img[masks[0]] = object_ids[0]
                mask_img = mask_img.numpy().astype(np.uint16)
                mask_img_pil = Image.fromarray(mask_img)
                mask_img_pil.save(os.path.join(temp_save_tracking_mask_results_dir, f"annotated_frame_{frame_idx:05d}.png"))

                if mask_img.max() != 0:
                    valid_frame_list.append(frame_idx)

            if object_name not in bbox_json_data[track_id]:
                bbox_json_data[track_id][object_name] = []

            bbox_json_data[track_id][object_name].extend(valid_frame_list)
            corresponding_json_data[track_id][object_name] = object_id
            control_json_data[track_id][object_name] = frame_idx


        # Save JSON files
        print("Saving JSON files...")
        with open(save_bbox_json_dir, 'w') as f:
            json.dump(bbox_json_data, f, indent=4)
        with open(save_corresponding_json_dir, 'w') as f:
            json.dump(corresponding_json_data, f, indent=4)
        with open(save_control_json_dir, 'w') as f:
            json.dump(control_json_data, f, indent=4)

        # Clean up temporary files
        print("Cleaning up temporary files...")
        shutil.rmtree(source_video_frame_dir)
        torch.cuda.empty_cache()
        gc.collect()

        print(f"Video {base_name} processing complete!")

    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return

def main():
    args = parse_args()

    # Check and download SAM2 model
    sam2_checkpoint = os.path.join(args.sam2_checkpoint_path, "sam2.1_hiera_large.pt")
    if not os.path.exists(sam2_checkpoint):
        print("Downloading SAM2 model from Hugging Face...")
        hf_hub_download(repo_id="facebook/sam2.1-hiera-large", filename="sam2.1_hiera_large.pt", local_dir=args.sam2_checkpoint_path)

    # Initialize model
    video_predictor = build_sam2_video_predictor(args.model_cfg, sam2_checkpoint)
    face_analyzer = FaceAnalysis(root="pretrained/face_encoder", providers=['CUDAExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(320, 320))

    # Get all video files
    video_file = args.video_folder
    

    # Process each video
    with torch.no_grad():
        process_single_video(video_file, args.output_path, video_predictor, face_analyzer)


if __name__ == "__main__":
    main() 