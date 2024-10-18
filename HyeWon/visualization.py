# visualization.py

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
from detectron2.structures import Boxes, Instances
import torch

def calculate_iou(gt_box, pred_box):
    """Calculates IoU between two bounding boxes."""
    x_left = max(gt_box[0], pred_box[0])
    y_top = max(gt_box[1], pred_box[1])
    x_right = min(gt_box[2], pred_box[2])
    y_bottom = min(gt_box[3], pred_box[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    union_area = gt_area + pred_area - intersection_area
    iou = intersection_area / union_area
    return iou

def visualize_predictions(image, predictions, annotations, filename, threshold=0.5):
    """Visualizes predictions with ground truth and IoU."""
    # Tensor를 numpy 배열로 변환
    if isinstance(image, torch.Tensor):
        if image.ndim == 3:  # C, H, W
            image = image.permute(1, 2, 0).cpu().numpy()
        elif image.ndim == 2:  # H, W (흑백 이미지 등)
            image = image.cpu().numpy()
        else:
            raise ValueError("Unexpected image shape: {}".format(image.shape))
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    pred_boxes = predictions.pred_boxes.tensor.cpu().numpy()
    pred_labels = predictions.pred_classes.cpu().numpy()
    pred_scores = predictions.scores.cpu().numpy()

    # Annotations가 None이 아닐 경우에만 처리
    if annotations is not None:
        gt_boxes = annotations.gt_boxes.tensor.cpu().numpy()
        gt_labels = annotations.gt_classes.cpu().numpy()

        for i, gt_box in enumerate(gt_boxes):
            best_iou = 0
            best_pred_idx = -1
            for j, pred_box in enumerate(pred_boxes):
                iou = calculate_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = j

            # Draw ground truth box
            gt_label = gt_labels[i]
            cv2.rectangle(image, (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])), (0, 0, 255), 2)
            cv2.putText(image, str(gt_label), (int(gt_box[0]), int(gt_box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Draw prediction box if IoU is below the threshold
            if best_pred_idx != -1 and best_iou < threshold:
                pred_box = pred_boxes[best_pred_idx]
                pred_label = pred_labels[best_pred_idx]
                pred_score = pred_scores[best_pred_idx]
                cv2.rectangle(image, (int(pred_box[0]), int(pred_box[1])), (int(pred_box[2]), int(pred_box[3])), (255, 0, 0), 2)
                cv2.putText(image, f"{pred_label}: {pred_score:.2f}, IoU: {best_iou:.2f}", (int(pred_box[0]), int(pred_box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imwrite(filename, image)
