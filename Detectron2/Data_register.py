# data_loader.py

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances


def register_datasets():
    """
    COCO 형식의 데이터셋 인스턴스를 등록합니다.
    """

    try:
        register_coco_instances('coco_trash_train', {}, '/data/ephemeral/home/baseline/detectron2/train_split.json',
                                '/data/ephemeral/home/dataset/')
    except AssertionError:
        pass

    try:
        register_coco_instances('coco_trash_test', {}, '/data/ephemeral/home/baseline/detectron2/val_split.json',
                                '/data/ephemeral/home/dataset/')
    except AssertionError:
        pass

    MetadataCatalog.get('coco_trash_train').thing_classes = ["General trash", "Paper", "Paper pack", "Metal",
                                                             "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery",
                                                             "Clothing"]
