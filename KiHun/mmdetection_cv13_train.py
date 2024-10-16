# 필요한 모듈 임포트
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmdet.utils import get_device

import wandb

def wandb_setup(cfg):
    # wandb
    wandb.login(key="350e50a8ed217678324bb18c08665de9ae70269a")

    cfg.log_config = dict(
        interval=50,
        hooks=[
            dict(type='TextLoggerHook'),
            dict(type='WandbLoggerHook',
                init_kwargs=dict(
                    project='mmdet_test',  
                    entity="superl3-naver",    
                    name='test',
                    tags=["MMDetection", "Cascade R-CNN", "No_Mosaic"],                    																             
                    group="model_test",
                    config={
                        "batch_size": cfg.data.samples_per_gpu,
                        "base_learning_rate": cfg.optimizer.lr,
                        #"max_epochs": cfg.runner.max_epochs,
                        "steps": cfg.lr_config.step,
                        "gamma": cfg.lr_config.gamma if hasattr(cfg.lr_config, 'gamma') else 0.1,
                        "model_weights": cfg.load_from,
                        "num_classes": 10,
                        "batch_size_per_image": cfg.model.roi_head.train_cfg.rcnn.sampler.num if hasattr(cfg.model.roi_head, 'train_cfg') else None
                    }            
                )
            )
        ]
    )
    print("wandb init done!")

    return cfg

def train(cfg):

    # 한 배치에 사용할 샘플 수 설정 -> 배치 사이즈
    cfg.data.samples_per_gpu = 4

    # 체크포인트 설정: 학습 도중 저장할 체크포인트 파일의 개수를 3개로 제한하고, 1 epoch마다 저장
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)

    # 학습을 실행할 디바이스 설정 (GPU/CPU 자동 감지)
    cfg.device = get_device()

    # 모델의 마지막 레이어 수정을 통해 10개의 클래스로 분류하도록 설정
    cfg.model.roi_head.bbox_head.num_classes = 10

    # 기울기 클리핑 설정 (학습 안정화를 위해 사용)
    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)

        # runner를 IterBasedRunner로 변경하여 스텝 단위 학습 수행
    cfg.runner = dict(type='IterBasedRunner', max_iters=3000)

    # 평가 주기(validation step)를 1500 스텝마다 실행하도록 설정
    cfg.evaluation = dict(interval=1500, metric='bbox')  # metric은 사용할 평가 기준

    # 체크포인트 저장 주기 설정 (예: 1500 스텝마다 체크포인트 저장)
    cfg.checkpoint_config = dict(interval=1500)
    
    # 학습 데이터셋 빌드
    # cfg.data.train 설정에 맞춰 학습 데이터셋을 생성합니다.
    datasets = [build_dataset(cfg.data.train)]
    print("dataset built!")

    # 데이터셋이 정상적으로 빌드되었는지 확인
    datasets[0]

    cfg = wandb_setup(cfg)

    # 모델 빌드 및 사전 학습된 가중치 로드
    # Faster R-CNN 모델을 빌드하고 가중치를 초기화합니다.
    model = build_detector(cfg.model)
    model.init_weights()

    # 모델 학습 시작
    # 빌드된 모델과 학습 데이터셋을 이용해 모델을 학습시킵니다.
    train_detector(model, datasets[0], cfg, distributed=False, validate=True)
