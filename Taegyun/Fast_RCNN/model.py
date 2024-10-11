# model.py

from detectron2.config import get_cfg
from detectron2 import model_zoo


def setup_model(config):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config['MODEL']['WEIGHTS']))
    cfg.DATASETS.TRAIN = (config['DATASETS']['TRAIN'],)
    cfg.DATASETS.TEST = (config['DATASETS']['TEST'],)
    cfg.DATALOADER.NUM_WORKERS = config['DATALOADER']['NUM_WORKERS']
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config['MODEL']['WEIGHTS'])
    cfg.SOLVER.IMS_PER_BATCH = config['SOLVER']['IMS_PER_BATCH']
    cfg.SOLVER.BASE_LR = config['SOLVER']['BASE_LR']
    cfg.SOLVER.MAX_ITER = config['SOLVER']['MAX_ITER']
    cfg.SOLVER.STEPS = tuple(config['SOLVER']['STEPS'])
    cfg.SOLVER.GAMMA = config['SOLVER']['GAMMA']
    cfg.SOLVER.CHECKPOINT_PERIOD = config['SOLVER']['CHECKPOINT_PERIOD']
    cfg.OUTPUT_DIR = config['OUTPUT_DIR']
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config['MODEL']['ROI_HEADS']['BATCH_SIZE_PER_IMAGE']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config['MODEL']['ROI_HEADS']['NUM_CLASSES']
    cfg.TEST.EVAL_PERIOD = config['TEST']['EVAL_PERIOD']

    return cfg
