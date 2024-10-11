# main.py

import os
import torch
import wandb
import yaml
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from data_loader import register_datasets
from Wandb import WandbLoggerHook
from train import MyTrainer
from datetime import datetime
from model import setup_model


def get_timestamp():
    now = datetime.now()
    return now.strftime("%m%d_%H%M")


# YAML 설정 파일 로드
def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# main 함수
def main():
    # YAML 설정 파일 로드
    config = load_config('Faster_rcnn_config.yaml')
    print('모델 세팅값을 가지고 왔습니다.')

    #Datset 등록
    register_datasets()
    print('데이터셋을 등록했습니다.')

    # 모델 설정
    cfg = setup_model(config)
    print('모델의 세팅값에 맞게 모델을 설정했습니다.')

    # 트레이너 초기화
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)

    wandb_logger_hook = WandbLoggerHook(
        model_name="Faster-RCNN",
        project_name="level2-objectdetection-cv-13",
        entity="superl3-naver",
        group="model_test",
        tags=["Detectron2", "Faster-R-CNN", "Non-Augmented"]
    )

    print('wandb을 시작합니다.')

    trainer.register_hooks([wandb_logger_hook])
    print('트레이너를 초기화 했습니다.')


    # Detectron2의 cfg 설정을 wandb.config로 자동 기록
    wandb.config.update({
        "batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "base_learning_rate": cfg.SOLVER.BASE_LR,
        "max_iterations": cfg.SOLVER.MAX_ITER,
        "steps": cfg.SOLVER.STEPS,
        "gamma": cfg.SOLVER.GAMMA,
        "model_weights": cfg.MODEL.WEIGHTS,
        "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
        "batch_size_per_image": cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
        "timestamp": f"{get_timestamp()}",
    })
    print('Wandb 자동기록을 설정했습니다.')


    # 모델 학습
    trainer.train()

    # WandB 저장
    wandb.config.update(cfg)


if __name__ == "__main__":
    main()
