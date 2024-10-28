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

### 실행
-config 파일을 통해서 사전 세팅
```실행
python main.py
```
