# train.py

import os
import torch
import wandb
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader
from detectron2.evaluation import COCOEvaluator
from Data_mapper import MyMapper
import time

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        return build_detection_train_loader(
            cfg, mapper=MyMapper, sampler=sampler
        )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs('./output_eval', exist_ok=True)
            output_folder = './output_eval'

        return COCOEvaluator(dataset_name, cfg, False, output_folder)
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self._data_loader_iter = iter(self.build_train_loader(cfg)) # _data_loader_iter 초기화
        self.accumulation_steps = cfg.SOLVER.ACCUMULATION_STEPS

    def run_step(self):
        """
        Run one iteration of training/evaluation.
        """
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        # Gradient Accumulation
        self.optimizer.zero_grad()
        with torch.no_grad():
            loss_dict = self.model(data)
            losses = self._get_loss(loss_dict, self.get_iter())
            loss_dict_reduced = {"loss": losses.item()}
            loss_dict_reduced_unscaled = {k: v.item() for k, v in loss_dict.items()}

        losses.backward()

        if (self.iteration + 1) % self.accumulation_steps == 0:
            self._run_optimizer_step(loss_dict) # 실제 optimizer step 실행
            self.scheduler.step() # lr scheduler step 실행

        self._write_metrics(loss_dict_reduced, data_time)
    
    def _get_loss(self, loss_dict, iteration): # _get_loss 메서드 추가
        losses = sum(loss_dict.values())
        assert torch.isfinite(losses).all(), loss_dict
        return losses
        
    def get_iter(self): # get_iter 메서드 추가
        return self.iteration