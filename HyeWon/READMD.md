# READMD.md

```aiignore
Faseter-RCNN
├── Faster_rcnn_config.yaml
├── Wandb.py
├── Data_register.py
├── Data_mapper.py
├── main.py
├── Model.py
└── trainer.py
```

### Faster_rcnn_config.yaml
- model config setting

### Data_register.py
- train, validation register
- Classes setting

### Model.py
- Model setting(config)

### Data_mapper.py
-  Data augmentation

### trainer.py
-  build_train_loader
-  build_evaluator

### Wandb.py
- Automatic recording

### main.py
- Faster_rcnn_config으로 모델 세팅값 설정




## 가설

### ROI가 증가하면 정확도가 올라갈 것이다.
```aiignore
DATASETS:
  TRAIN: 'coco_trash_train'
  TEST: 'coco_trash_test'

DATALOADER:
  NUM_WORKERS: 2

MODEL:
  WEIGHTS: 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
  ROI_HEADS:
    BATCH_SIZE_PER_IMAGE: 128
    NUM_CLASSES: 10

SOLVER:
  IMS_PER_BATCH: 4
  BASE_LR: 0.0025
  MAX_ITER: 15000
  STEPS: [8000, 12000]
  GAMMA: 0.005
  CHECKPOINT_PERIOD: 3000
  WARMUP_ITERS : 3000
OUTPUT_DIR: './output'
TEST:
  EVAL_PERIOD: 3000
```