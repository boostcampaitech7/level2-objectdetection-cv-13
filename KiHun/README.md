# mmdetection_old README
## 파일 목록

```
mmdetection
├─────config/~
├── mmdetection_cv13_inference.py 
├── mmdetection_cv13_train.py 
├── mmdetection_cv13_utils.py 
├── mmdetection_cv13.py 
└── mmdetection.ipynb
```
## 설명

-  **mmdetection_cv13_inference.py**
	-  inference를 실행하는데 필요한 코드
- **mmdetection_cv13_train.py**
	- train을 실행할때 필요한 코드
- **mmdetection_cv13_utils**
	- wandb 설정
	- 프로젝트용 dataset 설정
	-  프로그램 실행 시 인자에 대한 처리
- **mmdetection_cv13.py**
	-  ArgumentParser로 처리하기 위한 코드
- **mmdetection.ipynb**
	-  pycode화 시킨 코드를 사용하기 위한 주피터 노트북

## 사용방법

- python code화 시켰는데, 사용 편의성 (실행 기록)을 위해 주피터 노트북에서 실행


```
!python mmdetection_cv13.py --mode TRAIN/TEST --config {CONFIG_PATH}
```


# mmDetection README

## 파일 목록

```
mmdetection
├─────config/~
├── test_cv13.py 
├── train_cv13.py 
└── mmdetection.ipynb
```
## 설명

기본적으로 제공해주는 코드인 train.py / test.py를 프로젝트에 맞게 변형

-  **test_cv13.py**
	-  inference를 실행하는데 필요한 코드
- **train_cv13.py**
	- wandb에 데이터를 넘겨주기 위한 함수가 들어있음
	- train을 실행할때 필요한 코드
- **mmdetection.ipynb**
	-  pycode화 시킨 코드를 사용하기 위한 주피터 노트북

## 사용방법

- python code화 시켰는데, 사용 편의성 (실행 기록)을 위해 주피터 노트북에서 실행
- 기존 코드에서 제공하는 기능은 대부분 사용가능하나, 사용했던 파라미터나 추가한 파라미터에 대한 설명만 기재

### TRAIN

```
!python train_cv13.py {CONFIG_PATH} --amp --resume --checkdata
```

- **--amp**: auto mixed precision을 활성화하기 위한 인자, 일부 모델에선 사용 불가능
- **--resume**: 마지막 epoch에서 이어서 학습을 진행할 수 있게 해 주는 인자
- **--checkdata**: train시 증강이 어떤 식으로 보이는지 50장을 확인할 수 있게 해 주는 인자
### TEST

```
!python test_cv13.py {CONFIG_PATH} {CHECKPOINT_PATH} --skip --streamlit --tta
```

- **CHECKPOINT_PATH**: 원래와 같지만, 편의를 위해 work_dir/config_name의 경로를 자동 지정 해 주었음, 따라서 학습 후 사용 시 epoch_~.pth만 입력해서 inference할 수 있음
- **--skip**: test.py는 먼저 테스트를 하여 results.pkl을 만든 후 이 pkl파일 기반으로 제출용 csv 파일을 만드는데, inference 기능 구현 관련하여 테스트를 위해 pkl 파일 생성시 시간이 소요되기 때에 해당 test를 하는 과정을 생략하고 바로 pkl 기반으로 csv를 만드는 작업을 하기 위한 인자
- **--streamlit**: streamlit를 통해 비교하기 위해 만들어진 명령어, 수동으로 지정 해 준 경로에 {config_name}.csv로 validation dataset으로 테스트하여 생성
- **--tta**: tta를 적용하기 위한 인자

## TODO

- 배포를 고려하지 않고 작성했기 때문에, streamlit 경로, dataset 경로를 수동으로 직접 지정해 주었음, 이 점을 개선해야함