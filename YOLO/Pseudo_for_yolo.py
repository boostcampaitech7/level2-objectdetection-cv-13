import pandas as pd
import json
from datetime import datetime
import os
import shutil

def Pseudo(csv_file: str, save_dir: str, image_src_dir: str, image_dest_dir: str, conf_threshold: float = 0.9):
    '''
        csv_file: Pseudo Label을 만들 csv 파일 경로.
        save_dir: json 파일을 저장할 경로.
        image_src_dir: 원본 이미지들이 저장된 디렉토리.
        image_dest_dir: 복사될 이미지들의 저장될 디렉토리.
        conf_threshold: 신뢰도가 이 값 이상인 annotation만 사용.
    '''
    pseudo_labels_df = pd.read_csv(csv_file)
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [] 
    }

    categories = {
        0: "General trash",
        1: "Paper",
        2: "Paper pack",
        3: "Metal",
        4: "Glass",
        5: "Plastic",
        6: "Styrofoam",
        7: "Plastic bag",
        8: "Battery",
        9: "Clothing"
    }

    for category_id, category_name in categories.items():
        coco_output["categories"].append({
            "id": category_id,
            "name": category_name
        })

    annotation_id = 23144 
    image_id = 4883 

    for index, row in pseudo_labels_df.iterrows():
        image_name = row['image_id']  
        prediction_string = row['PredictionString']
        date_captured = datetime.now().isoformat()

        if not prediction_string:
            continue  

        predictions = prediction_string.split()
        valid_annotations = []

        for i in range(0, len(predictions), 6):
            label = int(predictions[i])
            score = float(predictions[i + 1])
            xmin = float(predictions[i + 2])
            ymin = float(predictions[i + 3])
            xmax = float(predictions[i + 4])
            ymax = float(predictions[i + 5])

            if score >= conf_threshold:
                w = xmax - xmin
                h = ymax - ymin
                valid_annotations.append({
                    "image_id": image_id, 
                    "category_id": label,
                    "bbox": [xmin, ymin, w, h],
                    "iscrowd": 0,
                    "id": annotation_id
                })
                annotation_id += 1  

        if valid_annotations:  
            src_image_path = os.path.join(image_src_dir, image_name)  

            if os.path.exists(src_image_path):
                dest_image_path = os.path.join(image_dest_dir, f'{image_id}.jpg')
                shutil.copy(src_image_path, dest_image_path)

            coco_output["images"].append({
                "file_name": f'{image_id}.jpg', 
                "width": 1024,
                "height": 1024,
                "license": 0,
                "flickr_url": None,
                "coco_url": None,
                "date_captured": date_captured,
                "id": image_id,
            })
            image_id += 1

            coco_output["annotations"].extend(valid_annotations)

    if os.path.exists(save_dir):
        raise FileExistsError(f"'{save_dir}'에 동일한 이름의 json 파일이 이미 존재")

    with open(save_dir, 'w') as json_file:
        json.dump(coco_output, json_file, indent=4)