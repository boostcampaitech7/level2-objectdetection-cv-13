from argparse import ArgumentParser

from mmdetection_cv13_utils import (set_default_dataset, load_config_from_arg, get_work_dir_from_arg, wandb_setup)
from mmdetection_cv13_train import train
from mmdetection_cv13_inference import inference

def main(args):
    # config 파일 로드
    cfg = load_config_from_arg(args)
    cfg.work_dir = get_work_dir_from_arg(args)
    
    # 기본 데이터셋 및 설정 적용
    cfg = set_default_dataset(cfg)
    
    cfg = wandb_setup(cfg, args.config)

    # 체크포인트 불러오기 옵션
    if args.checkpoint:
        cfg.load_from = args.checkpoint

    # 학습 또는 추론 모드 선택
    if args.mode == 'train':

        #if args.fp16 == True:
        #    cfg.fp16 = dict(loss_scale=512.0)

        train(cfg)

    elif args.mode == 'inference':
        inference(cfg)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference"],
                        help="모드를 선택하세요: train or inference")
    parser.add_argument("--checkpoint", type=str, default=None, help="체크포인트 경로")

    parser.add_argument("--config", type=str, default="faster_rcnn/faster_rcnn_r50_fpn_1x_coco", help="설정파일 경로, config 다음 경로를 확장자 없이 입력해주세요")

    parser.add_argument("--fp16", type=bool, default=True)
    
    args = parser.parse_args()
    main(args)