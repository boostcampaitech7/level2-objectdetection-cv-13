# data_mapper.py

import copy
import torch
import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils


def MyMapper(dataset_dict):
    """
    데이터셋을 변환 및 전처리합니다.
    mapper - input data를 어떤 형식으로 return할지 (따라서 augmnentation 등 데이터 전처리 포함 됨)
    
    Args:
        dataset_dict (dict): Detectron2 데이터셋 딕셔너리.

    Returns:
        dict: 변환된 데이터셋 딕셔너리.
    """
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')

    transform_list = [
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3),
        T.RandomRotation(angle=[-30, 30]),  # 랜덤 회전
    ]

    image, transforms = T.apply_transform_gens(transform_list, image)

    dataset_dict['image'] = torch.as_tensor(image.transpose(2, 0, 1).astype('float32'))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop('annotations')
        if obj.get('iscrowd', 0) == 0
    ]

    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict['instances'] = utils.filter_empty_instances(instances)

    return dataset_dict