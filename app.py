import subprocess
import re
from datetime import datetime
import os
from typing import List, Tuple, Optional

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from moviepy.editor import ImageSequenceClip
from sam2.build_sam import build_sam2_video_predictor

# Define the command to be executed
command = ["python", "setup.py", "build_ext", "--inplace"]

# Execute the command
result = subprocess.run(command, capture_output=True, text=True)

def execute_command(command):
    result = subprocess.run(command, capture_output=True, text=True)
    print("Output:\n", result.stdout)
    print("Errors:\n", result.stderr)

    if result.returncode == 0:
        print("Command executed successfully.")
    else:
        print("Command failed with return code:", result.returncode)

def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps

def clear_points(image):
    return [
        image,
        [],
        [],
        image,
    ]

def preprocess_video_in(video_path):
    unique_id = datetime.now().strftime('%Y%m%d%H%M%S')
    extracted_frames_output_dir = f'frames_{unique_id}'
    os.makedirs(extracted_frames_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(fps * 10)
    frame_number = 0
    first_frame = None

    while True:
        ret, frame = cap.read()
        if not ret or frame_number >= max_frames:
            break

        frame_filename = os.path.join(extracted_frames_output_dir, f'{frame_number:05d}.jpg')
        cv2.imwrite(frame_filename, frame)
        if frame_number == 0:
            first_frame = frame_filename
        frame_number += 1

    cap.release()
    scanned_frames = [p for p in os.listdir(extracted_frames_output_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]]
    scanned_frames.sort(key=lambda p: int(os.path.splitext(p)[0]))

    return first_frame, [], [], first_frame, first_frame, extracted_frames_output_dir, scanned_frames, None, None, False

def get_point(point_type, tracking_points, trackings_input_label, input_first_frame_image, evt):
    tracking_points.append(evt['index'])
    trackings_input_label.append(1 if point_type == "include" else 0)

    transparent_background = Image.open(input_first_frame_image).convert('RGBA')
    w, h = transparent_background.size
    radius = int(0.02 * min(w, h))
    transparent_layer = np.zeros((h, w, 4), dtype=np.uint8)

    for index, track in enumerate(tracking_points):
        color = (0, 255, 0, 255) if trackings_input_label[index] == 1 else (255, 0, 0, 255)
        cv2.circle(transparent_layer, track, radius, color, -1)

    transparent_layer = Image.fromarray(transparent_layer, 'RGBA')
    selected_point_map = Image.alpha_composite(transparent_background, transparent_layer)
    return tracking_points, trackings_input_label, selected_point_map

def initialize_torch():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def show_mask(mask, ax, obj_id=None, random_color=False):
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) if random_color else np.array([*plt.get_cmap("tab10")(obj_id)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], pos_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def load_model(checkpoint):
    if checkpoint == "tiny":
        return "./checkpoints/sam2_hiera_tiny.pt", "sam2_hiera_t.yaml"
    elif checkpoint == "small":
        return "./checkpoints/sam2_hiera_small.pt", "sam2_hiera_s.yaml"
    elif checkpoint == "base-plus":
        return "./checkpoints/sam2_hiera_base_plus.pt", "sam2_hiera_b+.yaml"
    elif checkpoint == "large":
        return "./checkpoints/sam2_hiera_large.pt", "sam2_hiera_l.yaml"

def get_mask_sam_process(stored_inference_state, input_first_frame_image, checkpoint, tracking_points, trackings_input_label, video_frames_dir, scanned_frames, working_frame=None, available_frames_to_check=[]):
    sam2_checkpoint, model_cfg = load_model(checkpoint)
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    video_dir = video_frames_dir
    frame_names = scanned_frames

    if stored_inference_state is None:
        inference_state = predictor.init_state(video_path=video_dir)
    else:
        inference_state = stored_inference_state

    ann_frame_idx = 0 if working_frame is None else int(re.search(r'frame_(\d+)', working_frame).group(1))
    ann_obj_id = 1
    points = np.array(tracking_points, dtype=np.float32)
    labels = np.array(trackings_input_label, np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points(inference_state=inference_state, frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels)

    plt.figure(figsize=(12, 8))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
    show_points(points, labels, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
    first_frame_output_filename = "output_first_frame.jpg"
    plt.savefig(first_frame_output_filename, format='jpg')
    plt.close()
    torch.cuda.empty_cache()

    if working_frame not in available_frames_to_check:
        available_frames_to_check.append(working_frame)
    return "output_first_frame.jpg", frame_names, predictor, inference_state, available_frames_to_check

def propagate_to_all(video_in, checkpoint, stored_inference_state, stored_frame_names, video_frames_dir, vis_frame_type, available_frames_to_check, working_frame):
    sam2_checkpoint, model_cfg = load_model(checkpoint)
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    inference_state = stored_inference_state
    frame_names = stored_frame_names
    video_dir = video_frames_dir

    frames_output_dir = "frames_output_images"
    os.makedirs(frames_output_dir, exist_ok=True)
    jpeg_images = []

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)}

    vis_frame_stride = 15 if vis_frame_type == "check" else 1

    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        output_filename = os.path.join(frames_output_dir, f"frame_{out_frame_idx}.jpg")
        plt.savefig(output_filename, format='jpg')
        plt.close()
        jpeg_images.append(output_filename)
        if f"frame_{out_frame_idx}.jpg" not in available_frames_to_check:
            available_frames_to_check.append(f"frame_{out_frame_idx}.jpg")
    
    torch.cuda.empty_cache()

    if vis_frame_type == "render":
        original_fps = get_video_fps(video_in)
        fps = original_fps
        clip = ImageSequenceClip(jpeg_images, fps=fps)
        final_vid_output_path = "output_video.mp4"
        clip.write_videofile(final_vid_output_path, codec='libx264')
        return final_vid_output_path, available_frames_to_check
    else:
        return jpeg_images, available_frames_to_check

def reset_propagation(first_frame_path, predictor, stored_inference_state):
    predictor.reset_state(stored_inference_state)
    return first_frame_path, [], [], stored_inference_state, ["frame_0.jpg"], first_frame_path

# Example of using the functions:

initialize_torch()

video_path = "path/to/video.mp4"
preprocessed_video = preprocess_video_in(video_path)
first_frame_path, tracking_points, trackings_input_label, first_frame, points_map, video_frames_dir, scanned_frames, stored_inference_state, stored_frame_names, video_in_drawer = preprocessed_video

evt = {'index': (100, 100), 'value': 'value', 'target': 'target'}
point_type = "include"
tracking_points, trackings_input_label, selected_point_map = get_point(point_type, tracking_points, trackings_input_label, first_frame, evt)

checkpoint = "tiny"
mask_result = get_mask_sam_process(stored_inference_state, first_frame, checkpoint, tracking_points, trackings_input_label, video_frames_dir, scanned_frames)
output_result, frame_names, predictor, inference_state, available_frames_to_check = mask_result

vis_frame_type = "check"
propagate_result = propagate_to_all(video_path, checkpoint, inference_state, frame_names, video_frames_dir, vis_frame_type, available_frames_to_check, "frame_0.jpg")
if vis_frame_type == "render":
    final_vid_output_path, available_frames_to_check = propagate_result
else:
    jpeg_images, available_frames_to_check = propagate_result

reset_result = reset_propagation(first_frame, predictor, inference_state)
first_frame_path, tracking_points, trackings_input_label, inference_state, scanned_frames, first_frame = reset_result
