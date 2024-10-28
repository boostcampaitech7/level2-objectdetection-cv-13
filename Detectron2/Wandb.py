# wandb_logger.py

from detectron2.engine.hooks import HookBase
from detectron2.utils.events import get_event_storage
import wandb
import datetime


def get_timestamp():
    now = datetime.datetime.now()
    return now.strftime("%m%d_%H%M")

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config




class WandbLoggerHook(HookBase):
    def __init__(self, model_name, project_name, entity, group, tags=None):
        super().__init__()

        self._window_size = 20

        # 로그인 및 초기화
        wandb.login(key="9e7b0afa953e262fd0c7cf990495712b40e0fb8c")  # 실제 API 키로 대체
        wandb.init(
            project=project_name,
            entity=entity,
            name=f"{model_name}_{get_timestamp()}",
            tags=tags if tags is not None else [],
            group=group,
        )

    def after_step(self):
        storage = get_event_storage()
        sendDict = self._makeStorageDict(storage)
        wandb.log(sendDict)

    def _makeStorageDict(self, storage):
        storageDict = {}
        for k, v in [(k, f"{v.median(self._window_size):.4g}") for k, v in storage.histories().items()]:
            storageDict[k] = float(v)

        return storageDict

# 사용할 때 객체 초기화 예시
# wandbLoggerHook = WandbLoggerHook(model_name="Faster-RCNN", project_name="level2-objectdetection-cv-13", entity="superl3-naver", group="model_test", tags=["Detectron2", "Faster-R-CNN", "Non-Augmented"])