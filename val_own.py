import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


def read_labels(file_path):
    with open(file_path, 'r') as f:
        return [line.strip().split() for line in f]

def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size  # returns (width, height)

def calculate_iou(box1, box2, image_width, image_height):
    # Convert normalized coordinates to pixel coordinates
    def normalize_to_pixel(box):
        return [
            float(box[1]) * image_width - (float(box[3]) * image_width) / 2,
            float(box[2]) * image_height - (float(box[4]) * image_height) / 2,
            float(box[1]) * image_width + (float(box[3]) * image_width) / 2,
            float(box[2]) * image_height + (float(box[4]) * image_height) / 2
        ]
    
    box1 = normalize_to_pixel(box1)
    box2 = normalize_to_pixel(box2)
    
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def calculate_metrics_with_confusion_matrix(pred_dir, truth_dir, image_dir, iou_threshold=0.1):
    pred_files = glob.glob(os.path.join(pred_dir, '*.txt'))
    tp = fp = fn = 0
    
    for pred_file in pred_files:
        base_name = os.path.basename(pred_file)
        image_name = base_name.replace('.txt', '.png')
        truth_file = os.path.join(truth_dir, base_name)
        image_file = os.path.join(image_dir, image_name)
        
        if not os.path.exists(truth_file):
            print(f"Warning: No corresponding truth file for {base_name}")
            continue
        
        if not os.path.exists(image_file):
            print(f"Warning: No corresponding image file for {image_name}")
            continue
        
        image_width, image_height = get_image_size(image_file)
        
        pred_boxes = read_labels(pred_file)
        truth_boxes = read_labels(truth_file)
        
        matched_truth_boxes = []
        
        for pred_box in pred_boxes:
            matched = False
            for i, truth_box in enumerate(truth_boxes):
                if i in matched_truth_boxes:
                    continue
                if calculate_iou(pred_box, truth_box, image_width, image_height) >= iou_threshold:
                    tp += 1
                    matched_truth_boxes.append(i)
                    matched = True
                    break
            if not matched:
                fp += 1
        
        fn += len(truth_boxes) - len(matched_truth_boxes)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print("tp", tp)
    print("fp", fp)
    print("fn", fn)
    print("tp + fp", tp + fp)
    print("tp + fn", tp + fn)
    return precision, recall, f1_score


# Directories
pred_dir = "/mnt/storage/ji/yolov5_2/runs/val/ex3.1class.focal_loss.iou0.0001.conf0.0001.expansion/labels_expanded"
truth_dir = "/mnt/storage/ji/Cerebral_Microbleeds_Dataset/dataset_CMB_1class/labels/val"
image_dir = "/mnt/storage/ji/Cerebral_Microbleeds_Dataset/dataset_CMB_1class/images/val"

precision, recall, f1_score, conf_matrix = calculate_metrics_with_confusion_matrix(pred_dir, truth_dir, image_dir)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1_score:.4f}")

# Plot and save confusion matrix
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Add text annotations for TP, FP, FN, TN
plt.text(-0.3, 0.5, 'Actual Positive', rotation=90, verticalalignment='center')
plt.text(-0.3, 1.5, 'Actual Negative', rotation=90, verticalalignment='center')
plt.text(0.5, -0.1, 'Predicted Positive', horizontalalignment='center')
plt.text(1.5, -0.1, 'Predicted Negative', horizontalalignment='center')
plt.text(0.5, 0.5, 'TP', horizontalalignment='center', verticalalignment='center')
plt.text(1.5, 0.5, 'FP', horizontalalignment='center', verticalalignment='center')
plt.text(0.5, 1.5, 'FN', horizontalalignment='center', verticalalignment='center')
plt.text(1.5, 1.5, 'TN', horizontalalignment='center', verticalalignment='center')

# Save the plot as a PNG file
output_dir = "/mnt/storage/ji/yolov5_2/runs/val/ex3.1class.focal_loss.iou0.0001.conf0.0001.expansion"
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()  # Close the plot to free up memory

print(f"Confusion matrix saved as 'confusion_matrix.png' in {output_dir}")