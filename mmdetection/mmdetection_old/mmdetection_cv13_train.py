# 필요한 모듈 임포트
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmdet.utils import get_device

def train(cfg):

    # 한 배치에 사용할 샘플 수 설정 -> 배치 사이즈
    cfg.data.samples_per_gpu = 4

    # 체크포인트 설정: 학습 도중 저장할 체크포인트 파일의 개수를 3개로 제한하고, 1 epoch마다 저장
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)

    # 학습을 실행할 디바이스 설정 (GPU/CPU 자동 감지)
    cfg.device = get_device()
    
    # 기울기 클리핑 설정 (학습 안정화를 위해 사용)
    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)

    # Set to iter-based runner and configure evaluation to run every 1500 steps
    #cfg.runner = dict(type='IterBasedRunner', max_iters=3000)

    # Set evaluation to run every 1500 steps
    #cfg.evaluation = dict(interval=1500, metric='bbox', save_best='bbox_mAP')

    # Checkpoint saving is also every 1500 steps
    #cfg.checkpoint_config = dict(interval=1500)

    cfg.runner = dict(type='EpochBasedRunner', max_epochs=3)
    
    # 학습 데이터셋 빌드
    # cfg.data.train 설정에 맞춰 학습 데이터셋을 생성합니다.
    datasets = [build_dataset(cfg.data.train)]
    print("dataset built!")

    # 데이터셋이 정상적으로 빌드되었는지 확인
    datasets[0]

    # 모델 빌드 및 사전 학습된 가중치 로드
    # Faster R-CNN 모델을 빌드하고 가중치를 초기화합니다.
    model = build_detector(cfg.model)
    model.init_weights()

    # 모델 학습 시작
    # 빌드된 모델과 학습 데이터셋을 이용해 모델을 학습시킵니다.
    train_detector(model, datasets[0], cfg, distributed=False, validate=True)
