import os
from mmcv import Config

def set_default_dataset(cfg):

    # 분류할 클래스 정의 (10개 클래스)
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    
    # 데이터셋의 루트 경로 설정
    root = '../../dataset/'

    # 데이터셋 관련 설정 수정
    # 학습 데이터셋의 클래스 및 이미지 경로, 어노테이션 파일 경로를 설정
    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = root
    cfg.data.train.ann_file = root + 'train_split.json'  # 학습 데이터셋 어노테이션 파일 경로 설정
    cfg.data.train.pipeline[2]['img_scale'] = (512,512)  # 이미지 크기를 512x512로 리사이즈

    cfg.data.val.classes = classes
    cfg.data.val.img_prefix = root  # 이미지 경로
    cfg.data.val.ann_file = root + 'val_split.json'  # 학습 데이터셋 어노테이션 파일 경로 설정
    cfg.data.val.pipeline[1]['img_scale'] = (512, 512)
    cfg.data.val.pipeline = cfg.data.test.pipeline # 이미지 크기를 512x512로 리사이즈

    # 테스트 데이터셋 설정
    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json'  # 테스트 데이터셋 어노테이션 파일 경로 설정
    cfg.data.test.pipeline[1]['img_scale'] = (512,512)  # 테스트 이미지 리사이즈 설정

    # 기타 학습 설정들
    cfg.seed = 2021  # 시드 설정 (재현성 보장)
    cfg.gpu_ids = [0]  # 사용할 GPU ID
    #cfg.work_dir = './work_dirs/faster_rcnn_r50_fpn_1x_trash'  # 작업 디렉터리 설정

    return cfg

def load_config_from_arg(args):

    # 기본 config 파일 경로
    base_config_dir = './configs'

    # 입력받은 config 파일 이름을 이용하여 전체 경로 생성
    config_file = os.path.join(base_config_dir, args.config + '.py')

    print("config path: " + config_file)

    # Config 객체 생성
    cfg = Config.fromfile(config_file)

    return cfg

def get_work_dir_from_arg(args):
    
    # 작업 디렉토리 설정
    work_dir = os.path.join('./work_dirs', args.config.replace('/','_'))

    print("work_dir path: " + work_dir)

    return work_dir