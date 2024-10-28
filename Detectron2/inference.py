# inference.py

import os
import torch
import yaml
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from Data_register import register_datasets
from visualization import visualize_predictions
import pandas as pd
from detectron2.data import build_detection_test_loader
from tqdm import tqdm
import numpy as np
from datetime import datetime

def get_timestamp():
    now = datetime.now()
    return now.strftime("%m%d_%H%M")

# YAML 설정 파일 로드
def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    config = load_config('Retina_Net_config.yaml')
    register_datasets()

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config['MODEL']['WEIGHTS']))
    cfg.MODEL.WEIGHTS = os.path.join(config['OUTPUT_DIR'], "model_final.pth")
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    test_loader = build_detection_test_loader(cfg, "coco_trash_test")

    visualization_dir = os.path.join(config['OUTPUT_DIR'], "visualization")
    os.makedirs(visualization_dir, exist_ok=True)
    max_visualizations_per_image = 5 

    for i, data in enumerate(tqdm(iter(test_loader))):
        try:
            data = data[0]
            image = data['image']
            annotations = data.get('instances', None)
            outputs = predictor(image)['instances']

            if outputs is not None and outputs.has("pred_boxes"):  # Check if outputs are valid
                scores = outputs.scores.cpu().numpy()
                keep_indices = scores > 0.5
                if np.any(keep_indices):
                    outputs = outputs[keep_indices]
                    num_visualizations = min(len(outputs), max_visualizations_per_image)
                    # num_visualizations 사용하여 시각화
                    if len(outputs) > 0:
                        filename = os.path.join(visualization_dir, f"visualization_{i}.png")
                        # annotations가 None인 경우에도 시각화 수행
                        visualize_predictions(image, outputs[:num_visualizations], annotations, filename, threshold=0.5)
        except Exception as e:
            print(f"Error during visualization of image {i}: {e}")

if __name__ == "__main__":
    main()
