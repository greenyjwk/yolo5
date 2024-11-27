import os
import glob
from typing import List, Dict, Tuple
from PIL import Image

def read_yolo_txt(file_path: str) -> List[List[float]]:
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [list(map(float, line.strip().split())) for line in lines]

def write_yolo_txt(file_path: str, bboxes: List[List[float]]):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        for bbox in bboxes:
            # Convert class ID to integer
            bbox_str = f"{int(bbox[0])} " + " ".join(f"{x:.6f}" for x in bbox[1:])
            f.write(bbox_str + '\n')

def parse_filename(file_name: str) -> Tuple[str, int]:
    parts = file_name.split('_')
    subject = parts[0].split('-')[1]
    slice_num = int(parts[2].split('.')[0])
    return subject, slice_num

def get_image_dimensions(png_path: str) -> Tuple[int, int]:
    with Image.open(png_path) as img:
        return img.size

def yolo_to_pixel_coords(bbox: List[float], img_width: int, img_height: int) -> List[int]:
    class_id, x_center, y_center, width, height, confidence = bbox
    x1 = int((x_center - width/2) * img_width)
    y1 = int((y_center - height/2) * img_height)
    x2 = int((x_center + width/2) * img_width)
    y2 = int((y_center + height/2) * img_height)
    return [x1, y1, x2, y2]

def iou(box1: List[int], box2: List[int]) -> float:
    eps = 1e-5
    """Calculate IoU between two bounding boxes in pixel coordinates."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = intersection /  (float(box1_area + box2_area - intersection) + eps)
    return iou

def group_adjacent_bboxes(subjects: Dict[str, Dict[int, List[List[float]]]], png_dir: str) -> Dict[str, List[List[Tuple[int, List[float]]]]]:
    grouped_bboxes = {}
    
    for subject, slices in subjects.items():
        grouped_bboxes[subject] = []
        slice_nums = sorted(slices.keys())
        
        for i, slice_num in enumerate(slice_nums):
            png_path = os.path.join(png_dir, f'sub-{subject}_slice_{slice_num:03d}.png')
            img_width, img_height = get_image_dimensions(png_path)
            
            for bbox in slices[slice_num]:
                pixel_bbox = yolo_to_pixel_coords(bbox, img_width, img_height)
                
                if not grouped_bboxes[subject] or i == 0:
                    grouped_bboxes[subject].append([(slice_num, bbox, pixel_bbox)])
                else:
                    prev_slice_num = slice_nums[i-1]
                    prev_png_path = os.path.join(png_dir, f'sub-{subject}_slice_{prev_slice_num:03d}.png')
                    prev_img_width, prev_img_height = get_image_dimensions(prev_png_path)
                    
                    matched = False
                    for group in grouped_bboxes[subject]:
                        if group[-1][0] == prev_slice_num:
                            prev_pixel_bbox = yolo_to_pixel_coords(group[-1][1], prev_img_width, prev_img_height)
                            if iou(pixel_bbox, prev_pixel_bbox) > 0.5:  # You can adjust this threshold
                                group.append((slice_num, bbox, pixel_bbox))
                                matched = True
                                break
                    
                    if not matched:
                        grouped_bboxes[subject].append([(slice_num, bbox, pixel_bbox)])
    
    return grouped_bboxes

def extend_voxels(input_dir: str, output_dir: str, png_dir: str):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"The {output_dir} is created") 
    
    txt_files = glob.glob(os.path.join(input_dir, '*.txt'))

    subjects: Dict[str, Dict[int, List[List[float]]]] = {}

    for txt_file in txt_files:
        file_name = os.path.basename(txt_file)
        subject, slice_num = parse_filename(file_name)
        
        if subject not in subjects:
            subjects[subject] = {}
        
        subjects[subject][slice_num] = read_yolo_txt(txt_file)
    
    grouped_bboxes = group_adjacent_bboxes(subjects, png_dir)
    
    # Initialize extended_bboxes with original bounding boxes
    extended_bboxes = {subject: {slice_num: bboxes.copy() for slice_num, bboxes in slices.items()} for subject, slices in subjects.items()}
    
    for subject, groups in grouped_bboxes.items():
        for group in groups:
            first_slice, first_bbox, _ = group[0]
            last_slice, last_bbox, _ = group[-1]
            
            # Extend to previous slice
            prev_slice = first_slice - 1
            if prev_slice not in extended_bboxes[subject]:
                extended_bboxes[subject][prev_slice] = []
            extended_bboxes[subject][prev_slice].append(first_bbox)
            
            # Extend to next slice
            next_slice = last_slice + 1
            if next_slice not in extended_bboxes[subject]:
                extended_bboxes[subject][next_slice] = []
            extended_bboxes[subject][next_slice].append(last_bbox)
    
    # Write the extended bounding boxes to the output directory
    for subject, slices in extended_bboxes.items():
        for slice_num, bboxes in slices.items():
            output_file = os.path.join(output_dir, f'sub-{subject}_slice_{slice_num:03d}.txt')
            write_yolo_txt(output_file, bboxes)

# Usage
input_dir = '/mnt/storage/ji/yolov5_2/runs/val/ex3.1class.focal_loss.iou0.0001.conf0.0001.expansion/labels'
output_dir = '/mnt/storage/ji/yolov5_2/runs/val/ex3.1class.focal_loss.iou0.0001.conf0.0001.expansion/labels_expanded'
png_dir = '/mnt/storage/ji/Cerebral_Microbleeds_Dataset/dataset_CMB_1class/images/val'
extend_voxels(input_dir, output_dir, png_dir)