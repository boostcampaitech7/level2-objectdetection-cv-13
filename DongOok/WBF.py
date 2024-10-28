from ensemble_boxes import *
import pandas as pd
import numpy as np
import os

def WBF(submission_df : list, save_dir):
    '''
        submission_df : WBF 앙상블을 진행할 submission(csv)들을 list로 지정해서 넣으면 된다.
        ex. submission_df = [submission1, submission2, submission3]
            ensemble_submission = WBF(submission_df, "home/submission/{저장할 이름}.csv")
    '''
    prediction_strings = []
    file_names = []
    iou_thr = 0.6 # 이걸 바꿔가면서 iou 임계값 지정
    image_ids = [f"test/{str(i).zfill(4)}.jpg" for i in range(4871)]

    for i, image_id in enumerate(image_ids):
        prediction_string = ''
        boxes_list = []
        scores_list = []
        labels_list = []

        for df in submission_df: 
            pred_df = df[df['image_id'] == image_id]

            if len(pred_df) > 0: # 해당 이미지에 대한 예측이 존재하는지 확인
                predict_string = pred_df['PredictionString'].tolist()[0]
                predict_list = str(predict_string).split()

                if len(predict_list) == 0:
                    continue

                predict_list = np.reshape(predict_list, (-1, 6))
                box_list = []

                # bbox 정규화(weighted_boxes_fusion에서 bbox는 0과 1사이어야함)
                for box in predict_list[:, 2:6].tolist():
                    box[0] = float(box[0]) / 1024
                    box[1] = float(box[1]) / 1024
                    box[2] = float(box[2]) / 1024
                    box[3] = float(box[3]) / 1024
                    box = np.clip(box, 0, 1)  # 각 box를 [0, 1] 범위로 클리핑(이거 없어도 잘 작동하는 warning이 떠서 추가함)
                    box_list.append(box)

                boxes_list.append(box_list)
                scores_list.append(list(map(float, predict_list[:, 1].tolist())))
                labels_list.append(list(map(int, predict_list[:, 0].tolist())))

        if len(boxes_list): # 여러 모델 중 하나라도 해당 이미지에서 예측을 했을 경우
            boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, iou_thr=iou_thr)
            
            # 해당 competition의 제출 형식에 맞게 변환
            for box, score, label in zip(boxes, scores, labels):
                if score < 0.05: # confidence score가 0.05미만 인 것들은 제외시킴
                    continue
                prediction_string += str(int(label)) + ' ' + str(score) + ' ' + str(box[0] * 1024) + ' ' + str(box[1] * 1024) + ' ' + str(box[2] * 1024) + ' ' + str(box[3] * 1024) + ' '

        prediction_strings.append(prediction_string)
        file_names.append(image_id)
        
    ens_df = pd.DataFrame({
        'PredictionString': prediction_strings,
        'image_id': file_names
    })

    if os.path.exists(save_dir):
        raise FileExistsError(f"Error: '{save_dir}'이 해당 경로에 동일한 이름으로 이미 존재")

    ens_df.to_csv(save_dir, index=False)
    return ens_df