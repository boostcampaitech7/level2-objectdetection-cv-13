from mmdet.datasets import (build_dataloader, build_dataset)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
import os
from mmcv.parallel import MMDataParallel
import pandas as pd
from pycocotools.coco import COCO

def inference(cfg):
    # 데이터셋의 루트 경로 및 마지막 학습된 epoch 설정
    epoch = 'latest'

    cfg.data.test.test_mode = True  # 테스트 모드로 설정
    # 테스트 모드이므로 학습 설정을 None으로 설정
    cfg.model.train_cfg = None
    
    # 한 배치에 처리할 샘플 수 설정 -> 배치 사이즈
    cfg.data.samples_per_gpu = 4

    # 모델의 최종 레이어 수정 (10개의 클래스로 분류)
    cfg.model.roi_head.bbox_head.num_classes = 10

    # 옵티마이저 설정에서 기울기 클리핑을 적용하여 학습 안정화
    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)


    # 데이터셋과 데이터로더를 빌드
    # 설정에 맞는 데이터셋 객체와 데이터로더 객체를 생성합니다.
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,  # 한 번에 1개의 샘플만 로드
        workers_per_gpu=cfg.data.workers_per_gpu,  # 워커 수
        dist=False,  # 분산 학습 비활성화
        shuffle=False  # 테스트 중이므로 셔플하지 않음
    )

    # 체크포인트 경로 설정
    checkpoint_path = os.path.join(cfg.work_dir, f'{epoch}.pth')

    # 모델 빌드 및 체크포인트 로드
    # Faster R-CNN 모델을 빌드하고, 저장된 체크포인트를 로드합니다.
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')  # CPU에서 로드

    # 모델의 클래스 정보를 데이터셋과 동일하게 설정
    model.CLASSES = dataset.CLASSES

    # GPU 병렬 처리 설정 (첫 번째 단일 GPU 사용)
    model = MMDataParallel(model.cuda(), device_ids=[0])

    # 모델을 이용한 테스트 실행
    # 설정된 데이터로더와 함께 모델 추론을 실행하고, 결과를 저장
    output = single_gpu_test(model, data_loader, show_score_thr=0.05)  # 0.05 이상의 confidence만 출력

    # COCO 어노테이션을 이용해 후처리 진행
    # output 결과를 submission 파일 형식에 맞게 변환
    prediction_strings = []
    file_names = []
    coco = COCO(cfg.data.test.ann_file)  # COCO 형식의 어노테이션 로드
    img_ids = coco.getImgIds()  # 이미지 ID를 가져옴

    class_num = 10  # 클래스 수 정의

    # 모델의 예측 결과를 후처리하여 각 이미지에 대해 PredictionString 생성
    for i, out in enumerate(output):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]  # 이미지 정보 로드
        for j in range(class_num):  # 각 클래스별로 예측된 결과에 대해
            for o in out[j]:
                # 클래스, confidence, bbox 정보를 submission 양식에 맞춰 기록
                prediction_string += str(j) + ' ' + str(o[4]) + ' ' + str(o[0]) + ' ' + str(o[1]) + ' ' + str(o[2]) + ' ' + str(o[3]) + ' '
        
        # 파일 이름과 함께 prediction_string을 저장
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])

    # 결과를 데이터프레임으로 저장 (submission.csv 파일 생성)
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(cfg.work_dir, f'submission_{epoch}.csv'), index=None)  # CSV 파일로 저장
    submission.head()  # 결과 미리보기
