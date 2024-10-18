# model.py

from detectron2.config import get_cfg
from detectron2 import model_zoo
import os


def setup_model(config):
    cfg = get_cfg()
    # 기본 구성 파일에서 모델 구성 불러오기
    # cfg.merge_from_file(model_zoo.get_config_file(config['MODEL']['WEIGHTS']))
    cfg_filename = config['MODEL']['WEIGHTS']
    if os.path.exists(cfg_filename): # os.path.exists 사용
        cfg.merge_from_file(cfg_filename)
    else:
        print(f"Warning: Config file '{cfg_filename}' does not exist! Using default config.")

    # 데이터셋 설정
    cfg.DATASETS.TRAIN = (config['DATASETS']['TRAIN'],)
    cfg.DATASETS.TEST = (config['DATASETS']['TEST'],)

    # 데이터 로더 설정
    cfg.DATALOADER.NUM_WORKERS = config['DATALOADER']['NUM_WORKERS']

    # 데이터 가중치 설정
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config['MODEL']['WEIGHTS'])

    # 솔버 설정
    cfg.SOLVER.IMS_PER_BATCH = config['SOLVER']['IMS_PER_BATCH']
    cfg.SOLVER.BASE_LR = config['SOLVER']['BASE_LR']
    cfg.SOLVER.MAX_ITER = config['SOLVER']['MAX_ITER']
    cfg.SOLVER.STEPS = tuple(config['SOLVER']['STEPS'])
    cfg.SOLVER.GAMMA = config['SOLVER']['GAMMA']
    cfg.SOLVER.CHECKPOINT_PERIOD = config['SOLVER']['CHECKPOINT_PERIOD']
    cfg.SOLVER.WARMUP_ITERS = config['SOLVER']['WARMUP_ITERS']
    # cfg.SOLVER.AMP.ENABLED = config['SOLVER']['AMP']['ENABLED']
    # TridentNet
    cfg.SOLVER.ACCUMULATION_STEPS = config['SOLVER']['ACCUMULATION_STEPS']
    cfg.SOLVER.AMP.ENABLED = config['SOLVER'].get('AMP', {}).get('ENABLED', False)

    # 츌룍 다랙토리 설정
    cfg.OUTPUT_DIR = config['OUTPUT_DIR']

    # ROI 헤드 설정
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config['MODEL']['ROI_HEADS']['BATCH_SIZE_PER_IMAGE']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config['MODEL']['ROI_HEADS']['NUM_CLASSES']
    # cfg.MODEL.RETINANET.NUM_CLASSES = config['MODEL']['RETINANET']['NUM_CLASSES'] # RetinaNet NUM_CLASSES 설정


    # 평가 주기 설정
    cfg.TEST.EVAL_PERIOD = config['TEST']['EVAL_PERIOD']

    return cfg
