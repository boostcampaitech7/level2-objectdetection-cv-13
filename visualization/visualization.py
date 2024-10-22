import os
import os.path as osp
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
from torchvision.utils import make_grid

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.visualization import LocalVisBackend, WandbVisBackend
from mmdet.utils import setup_cache_size_limit_of_dynamo

from mmengine.registry import DATASETS

def calculate_confusion_matrix(runner, outputs):
    """Confusion Matrix 계산 함수."""
    confusion_matrix = np.zeros((len(runner.data_loader.dataset.CLASSES), 
                                 len(runner.data_loader.dataset.CLASSES)))

    for batch_outputs in outputs:
        batch_results = batch_outputs['results']
        for i, result in enumerate(batch_results):
            gt_labels = batch_outputs['data_samples'][i]['gt_instances'].labels
            predicted_labels = [label for label in result.labels]

            for gt_label, predicted_label in zip(gt_labels, predicted_labels):
                confusion_matrix[gt_label, predicted_label] += 1

    return confusion_matrix


def plot_confusion_matrix(confusion_matrix, classes, title='Confusion Matrix'):
    """Confusion Matrix 시각화 함수."""
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f'
    thresh = confusion_matrix.max() / 2.
    for i, j in np.ndindex(confusion_matrix.shape):
        plt.text(
            j,
            i,
            format(confusion_matrix[i, j], fmt),
            horizontalalignment='center',
            color='white' if confusion_matrix[i, j] > thresh else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def visualize_confusion_matrix(runner, outputs):
    """Confusion Matrix를 시각화하는 함수."""

    # confusion matrix 계산
    confusion_matrix = calculate_confusion_matrix(runner, outputs)

    # confusion matrix 시각화
    plot_confusion_matrix(confusion_matrix, runner.data_loader.dataset.CLASSES)
    
    # 로컬에서 시각화 확인
    plt.show()

def visualize_predictions_vs_groundtruth(runner, outputs):
    """원본 이미지와 예측 결과를 비교하여 시각화하는 함수."""
    
    for batch_outputs in outputs:
        batch_results = batch_outputs['results']
        batch_data_samples = batch_outputs['data_samples']
        
        for i, result in enumerate(batch_results):
            # 원본 이미지 경로
            img_path = batch_data_samples[i].img_path
            # 이미지 로드
            img = plt.imread(img_path)
            
            # groundtruth annotation (bbox, label)
            gt_instances = batch_data_samples[i]['gt_instances']
            gt_bboxes = gt_instances.bboxes
            gt_labels = gt_instances.labels
            
            # 예측 결과 (bbox, label)
            predicted_bboxes = result.bboxes
            predicted_labels = result.labels

            # 시각화를 위한 이미지 생성
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(img)

            # groundtruth annotation 표시
            for bbox, label in zip(gt_bboxes, gt_labels):
                x1, y1, x2, y2 = bbox
                ax.add_patch(
                    Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        fill=False,
                        edgecolor='green',
                        linewidth=2))
                ax.text(
                    x1,
                    y1 - 5,
                    runner.data_loader.dataset.CLASSES[label],
                    bbox=dict(facecolor='green', alpha=0.5),
                    fontsize=10,
                    color='white')

            # 예측 결과 표시
            for bbox, label in zip(predicted_bboxes, predicted_labels):
                x1, y1, x2, y2 = bbox
                ax.add_patch(
                    Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        fill=False,
                        edgecolor='red',
                        linewidth=2))
                ax.text(
                    x1,
                    y1 - 5,
                    runner.data_loader.dataset.CLASSES[label],
                    bbox=dict(facecolor='red', alpha=0.5),
                    fontsize=10,
                    color='white')

            # 시각화 결과 저장
            ax.set_title(f'Groundtruth vs Prediction')
            plt.show()
            plt.close(fig)


def calculate_class_confusion(runner, outputs):
    """validation 결과를 기반으로 어떤 class를 다른 어떤 class랑 헷갈려 하는지 확인."""
    
    # confusion matrix 계산
    confusion_matrix = calculate_confusion_matrix(runner, outputs)
                
    # 각 class별로 가장 많이 헷갈리는 class 출력
    for i in range(len(runner.data_loader.dataset.CLASSES)):
        # 가장 많이 헷갈리는 class 찾기
        most_confused_class = np.argmax(confusion_matrix[i])
        if most_confused_class != i:
            print(f"Class '{runner.data_loader.dataset.CLASSES[i]}' is often confused with '{runner.data_loader.dataset.CLASSES[most_confused_class]}'")
            
    # Confusion matrix 비율로 변환
    # (행: 실제 클래스, 열: 예측 클래스)
    class_confusion_ratio = confusion_matrix.astype(np.float32) / np.sum(
        confusion_matrix, axis=1, keepdims=True)
    print(f"Class Confusion Ratio: {class_confusion_ratio}")
    
    # 추가 분석: 
    # 1. 특정 클래스의 오류 원인 분석:  헷갈리는 클래스의 이미지들을 살펴보고 오류 원인을 파악
    # 2. 데이터셋 불균형: Confusion matrix의 행별 합을 확인하여 데이터셋 불균형 여부 확인

def visualize_feature_maps(runner, outputs):
    """Feature Map을 시각화하는 함수."""

    # outputs: validation 결과 (batch 단위)
    # kwargs: 추가적인 인자 (예: epoch, iteration 등)

    for batch_outputs in outputs:
        batch_results = batch_outputs['results']
        batch_data_samples = batch_outputs['data_samples']

        for i, result in enumerate(batch_results):
            # 원본 이미지 경로
            img_path = batch_data_samples[i].img_path
            # 이미지 로드
            try:
                img = plt.imread(img_path)
            except FileNotFoundError:
                print(f"Error: Image file {img_path} not found.")
                continue

            # feature map 추출 (ResNet 모델의 경우)
            with torch.no_grad():
                feature_maps = []
                for layer_name, layer in runner.model.backbone.named_modules():
                    if isinstance(layer, torch.nn.Conv2d):
                        feature_maps.append(layer(batch_outputs['outputs']['features']))

            # feature map 시각화
            fig, axs = plt.subplots(len(feature_maps), 1, figsize=(10, 6 * len(feature_maps)))
            for j, feature_map in enumerate(feature_maps):
                feature_map = feature_map[0].cpu().detach().numpy()
                feature_map = np.mean(feature_map, axis=0)  # 채널 평균
                axs[j].imshow(feature_map, cmap='gray')
                axs[j].set_title(f'Feature Map {j+1}')
                axs[j].axis('off')

            # 시각화 결과 저장
            fig.suptitle(f'Feature Maps')
            plt.show()
            plt.close(fig)

def visualize_precision_recall_curve(runner, outputs):
    """Precision-Recall Curve를 시각화하는 함수."""

    # outputs: validation 결과 (batch 단위)
    # kwargs: 추가적인 인자 (예: epoch, iteration 등)

    from mmdet.evaluation.metrics import precision_recall_f1_score

    # precision-recall curve 계산
    precisions, recalls, f1_scores, thresholds = precision_recall_f1_score(outputs)

    # precision-recall curve 시각화
    plt.figure(figsize=(8, 6))
    for i in range(len(runner.data_loader.dataset.CLASSES)):
        plt.plot(recalls[i], precisions[i], label=runner.data_loader.dataset.CLASSES[i])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()