_base_ = [
    'deformable_detr_r50_16x2_50e_coco.py'
]
model = dict(
    bbox_head=dict(
        num_classes=10))