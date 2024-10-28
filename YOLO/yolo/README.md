### YOLO
- ultralytics 라이브러리를 기반으로 사용했습니다.
- https://github.com/ultralytics/ultralytics

### coco2yolo
- images 폴더 안에 각 train, val, test별로 이미지가 있습니다.
- labels 폴더 안에는 각 train, val, test별로 annotation 정보가 txt파일로 저장됩니다.

### rc_trash.yaml
- 모델 학습 시, 필요한 class를 정의하고, 각 이미지가 존재하는 파일 경로를 지정합니다.

### yolo.ipynb
- ultralytics으로 yolo모델을 사용한 코드입니다.