# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import matplotlib.pyplot as plt
from mmengine.dataset import Compose

import mmcv
from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo


from mmengine.registry import DATASETS

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--checkdata',
        action='store_true',
        default=False
    )
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

import datetime
import wandb
from mmengine.visualization import LocalVisBackend, WandbVisBackend

def get_timestamp():
    now = datetime.datetime.now()
    return now.strftime("%m%d_%H%M")

def wandb_setup(cfg, model_path):
    wandb.login(key="fe7040ef30d4610f369aab8d937183ce5e399c34")
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    print(model_name)
    vis_backends = [
            dict(type='WandbVisBackend',
                init_kwargs=dict(
                    project='mmdet_test',
                    entity="superl3-naver",    
                    name=f"{model_name}_{get_timestamp()}",
                    tags=["MMDetection", model_name],                     
                    group="model_test",
                    config={
                        "batch_size":cfg.train_dataloader.batch_size,
                        "num_workers":cfg.train_dataloader.num_workers,  
                    }            
                )
            )
        ]
    cfg.visualizer = dict(type='Visualizer', vis_backends=vis_backends, name="visualizer")  # wandb를 위한 비주얼라이저
    
    #cfg.log_level = 'DEBUG'
    
    print("wandb init done!")

    return cfg


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)

    cfg = wandb_setup(cfg, args.config)

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    if args.checkdata == True:
        # Runner가 설정된 후, 데이터셋에 접근하는 방법
        train_dataloader = runner.train_dataloader

        # DataLoader에서 Dataset 접근
        train_dataset = train_dataloader.dataset

        for i in range(50):
            # 첫 번째 샘플 확인 (증강 적용 후)
            sample = train_dataset[i]

            img = sample['inputs'].numpy().transpose(1, 2, 0)

            # 이미지 저장 (예: PNG 형식으로 저장)
            output_path = 'augmented_image' + str(i) + '.png'

            # 이미지가 BGR 형식이라면 RGB로 변환 (optional, 필요한 경우만)
            img_rgb = mmcv.bgr2rgb(img)

            # 이미지 파일로 저장 (PNG 형식)
            plt.imsave(output_path, img_rgb)

    if args.checkdata == False:
        # start training
        runner.train()


if __name__ == '__main__':
    main()
