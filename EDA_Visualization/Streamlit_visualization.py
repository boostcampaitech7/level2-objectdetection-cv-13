import json
import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="데이터 분석 및 개별 이미지 확인용", layout="wide")

global categories, colors1, colors2, output_directory, valid_directory
categories = ['General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
colors1 = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (165, 42, 42),  # Brown
    (255, 192, 203) # Pink
]
colors2 = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'pink']
output_directory = './output'
valid_directory = './valid'

# json 파일에서 각 key 별로 데이터 불러와서 dataframe으로 변환 후 리스트에 넣고 리스트 반환
# 입력 - json 파일
# 출력 - 데이터프레임 딕셔너리
def read_data_from_json_by_columns(filename):
    data = {}
    for key in filename:
        if type(filename[key]) == list:
            data[key] = pd.DataFrame(filename[key])
        else:
            data[key] = pd.DataFrame([filename[key]])
    return data

@st.cache_data
def load_json_data():
    with open('../dataset/train.json') as t:
        train_data = json.loads(t.read())
    with open('../dataset/val_split.json') as t:
        val_data = json.loads(t.read())
    with open('../dataset/test.json') as t:
        test_data = json.loads(t.read())
    test = read_data_from_json_by_columns(test_data)
    val = read_data_from_json_by_columns(val_data)
    train = read_data_from_json_by_columns(train_data)
    train['images']['annotation_num'] = train['annotations']['image_id'].value_counts()
    val['images']['annotation_num'] = val['annotations']['image_id'].value_counts()

    return test,val,  train, test_data, val_data, train_data
# 출력 - train, test 데이터프레임 딕셔너리

# 데이터 페이지 단위로 데이터프레임 스플릿
# 입력 - input_df(이미지 데이터), anno_df(박스 그리기 용), rows(한번에 보여줄 데이터 수)
# 출력 - df(이미지 데이터프레임 리스트), df2(박스 그리기 용 데이터프레임 리스트)
@st.cache_data()
def split_frame(input_df, rows):
    df = [input_df.loc[i : i + rows - 1, :] for i in range(0, len(input_df), rows)]
    return df

@st.cache_data()
def csv_to_dataframe(dir, csv_file):
    file_path = os.path.join(dir, csv_file)  # 파일 경로 생성
    df = pd.read_csv(file_path)  # csv 파일을 DataFrame으로 불러오기
    df['image_id'] = df['image_id'].str.extract(r"(\d+)").astype(int)
    annotation = []
    ann_id = 0
    
    # 각 파일의 내용 처리
    for row in df.itertuples(index=False, name=None):
        img = row[1]  # image_id
        pred_str = row[0]  # PredictionString

        # PredictionString이 NaN일 경우 스킵
        if pd.isna(pred_str):
            continue
        
        pred = list(map(float, pred_str.split()))
        
        # 예측값을 6개씩 묶어서 처리
        for j in range(0, len(pred), 6):
            if j + 5 >= len(pred):  # 인덱스 범위 체크
                continue
            
            category_id = int(pred[j])
            confidence = pred[j + 1]
            bbox = (pred[j + 2], pred[j + 3], pred[j + 4]-pred[j + 2], pred[j + 5]-pred[j + 3])  # (x, y, w, h)
            area = pred[j + 4] * pred[j + 5]  # 넓이 계산 (w * h)

            # annotation 리스트에 추가
            annotation.append({
                "image_id": img,
                "category_id": category_id,
                "area": area,
                "bbox": bbox,
                "isclowd": 0,
                "id": ann_id,
                "confidence": confidence
            })
            
            ann_id += 1
    
    anno = pd.DataFrame(annotation)
    
    return anno

def csv_list(output_dir):
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
    return csv_files
# 팝업창 띄우기

def check_same_csv(name, csv):
    i = 1
    while name in csv:
        if i == 1:
            name = name[:-4]+'_'+str(i)+'.csv'
        else:
            name = name[:-6]+'_'+str(i)+'.csv'
        i += 1
    return name

@st.dialog("csv upload")
def upload_csv(csv):
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    # 파일이 업로드되면 처리
    if uploaded_file is not None:
        # Pandas를 사용해 CSV 파일 읽기
        df = pd.read_csv(uploaded_file)
        df = df[['PredictionString','image_id']]

        # DataFrame 내용 출력
        st.write("Data Preview:")
        st.dataframe(df)

        input_name = st.text_input("csv 파일 이름 지정", value=uploaded_file.name.replace('.csv', ''))
        if st.button("upload_csv"):
            name = check_same_csv(input_name+'.csv',csv)
            st.write("saved file name: "+name)
            df.to_csv('./output/'+name,index=False)
        if st.button("close"):
            st.rerun()

def csv_to_backup(csv):
    os.rename('./output/'+csv,'./backup/'+csv)
    st.rerun()

# 페이지에 있는 이미지 출력
# 입력
## type = 이미지 경로 찾을 때 사용(../dataset/train/, ../dataset/test/ 이미지 서로 다른 폴더인 경우 사용 가능),
## img_pathes = train_data['image_id'] or test_data['image_id'] 데이터프레임
## anno = train_data[['bbox','category_id']] 데이터프레임
## window = 데이터 출력할 창
def get_image(image_path, anno, transform):
    img = cv2.imread(image_path)
    tlist = [0 for _ in range(10)]
    tset = set()

    if transform:
        transformed = transform(image=img, bboxes=anno['bbox'].tolist(), labels=anno['category_id'].tolist())
        img = transformed['image']
        anno = pd.DataFrame({'bbox': transformed['bboxes'], 'category_id': transformed['labels'], 'confidence': anno['confidence']})

    if not anno.empty:
        if 'confidence' in anno:
            iters = anno[['bbox','category_id','confidence']].values
            for annotation,trash,score in iters:
                if score<st.session_state['confidence']: continue
                cv2.rectangle(img, np.rint(annotation).astype(np.int32), colors1[trash], 3)
                ((text_width, text_height), _) = cv2.getTextSize(categories[trash], cv2.FONT_HERSHEY_SIMPLEX, 1, 10)
                cv2.rectangle(img, (int(annotation[0]), int(annotation[1]) - int(1.3 * text_height)), (int(annotation[0] + text_width), int(annotation[1])), colors1[trash], -1)
                cv2.putText(
                    img,
                    text=categories[trash],
                    org=(int(annotation[0]), int(annotation[1]) - int(0.3 * text_height)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, 
                    color=(0,0,0), 
                    lineType=cv2.LINE_AA,
                )
                tlist[trash] += 1
        else:
            iters = anno[['bbox','category_id']].values
            for annotation,trash in iters:
                cv2.rectangle(img, np.rint(annotation).astype(np.int32), colors1[trash], 3)
                ((text_width, text_height), _) = cv2.getTextSize(categories[trash], cv2.FONT_HERSHEY_SIMPLEX, 1, 10)
                cv2.rectangle(img, (int(annotation[0]), int(annotation[1]) - int(1.3 * text_height)), (int(annotation[0] + text_width), int(annotation[1])), colors1[trash], -1)
                cv2.putText(
                    img,
                    text=categories[trash],
                    org=(int(annotation[0]), int(annotation[1]) - int(0.3 * text_height)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, 
                    color=(0,0,0), 
                    lineType=cv2.LINE_AA,
                )
                tlist[trash] += 1
    for id, t in enumerate(tlist):
        if t:
            tset.add((categories[id],t))
    return img, tlist, tset

def show_images(type, img_pathes, anno, window):
    cols = window.columns(1)
    for idx,(path,id) in enumerate(img_pathes.values):
        if idx%1 == 0:
            cols = window.columns(1)
        if not anno.empty:
            img, tlist, tset = get_image(type+path, anno[anno['image_id']==id], 0)
        else:
            img, tlist, tset = get_image(type+path, pd.DataFrame(), 0)
        cols[idx%1].image(img)
        cols[idx%1].write(path)
        if tlist: 
            cols[idx%1].write(tset)

# 데이터 프레임 페이지 단위로 출력
# 입력
## img = train_data or test_data에서 'images'
## anno = train_data or test_data에서 'annotations'
## window = 데이터 프레임 출력할 위치
## type = 이미지 경로
def show_dataframe(img,anno,window,type):
    # 가장 윗부분 데이터 정렬할 지 선택, 정렬 시 무엇으로 정렬할지, 오름차순, 내림차순 선택
    top_menu = window.columns(3)
    with top_menu[0]:
        sort = st.radio("Sort Data", options=["Yes", "No"], horizontal=1, index=1, key=[type,window,1])
    if sort == "Yes":
        with top_menu[1]:
            sort_field = st.selectbox("Sort By", options=img.columns, key=[type,window,2])
        with top_menu[2]:
            sort_direction = st.radio(
                "Direction", options=["⬆️", "⬇️"], horizontal=True
            )
        img = img.sort_values(
            by=sort_field, ascending=sort_direction == "⬆️", ignore_index=True
        )



    # 데이터 크기 출력
    total_data = img.shape
    with top_menu[0]:
        st.write("data_shape: "+str(total_data))
    
    
    con1,con2 = window.columns((1,3))

    # 아래 부분 페이지당 데이터 수, 페이지 선택
    bottom_menu = window.columns((4, 1, 1))
    with bottom_menu[2]:
        batch_size = st.selectbox("Page Size", options=[9, 15, 27], key=[type,window,3])
    with bottom_menu[1]:
        total_pages = (
            int(len(img) / batch_size) if int(len(img) / batch_size) > 0 else 1
        )
        current_page = st.number_input(
            "Page", min_value=1, max_value=total_pages, step=1
        )
    with bottom_menu[0]:
        st.markdown(f"Page **{current_page}** of **{total_pages}** ")
    pages = split_frame(img, batch_size)
    
    
    
    if 'annotation_num' in pages[0].columns:
        con1.dataframe(data=pages[current_page - 1][['file_name','annotation_num']], use_container_width=True)
    else:
        con1.dataframe(data=pages[current_page - 1]['file_name'], use_container_width=True)
    if not anno.empty:
        show_images(type, pages[current_page - 1][['file_name','id']], anno, con2)
    else:
        show_images(type, pages[current_page - 1][['file_name','id']], pd.DataFrame(), con2)


def show_dataframe2(img,anno,window,type, anno1 = None):
    # 가장 윗부분 데이터 정렬할 지 선택, 정렬 시 무엇으로 정렬할지, 오름차순, 내림차순 선택
    top_menu = window.columns(3)
    with top_menu[0]:
        sort = st.radio("Sort Data", options=["Yes", "No"], horizontal=1, index=1, key=[type,window,1])
    if sort == "Yes":
        with top_menu[1]:
            sort_field = st.selectbox("Sort By", options=img.columns, key=[type,window,2])
        with top_menu[2]:
            sort_direction = st.radio(
                "Direction", options=["⬆️", "⬇️"], horizontal=True
            )
        img = img.sort_values(
            by=sort_field, ascending=sort_direction == "⬆️", ignore_index=True
        )




    # 데이터 크기 출력
    total_data = img.shape
    with top_menu[0]:
        st.write("data_shape: "+str(total_data))
    
    
    con1,con2, con3 = window.columns((1,2,2))

    # 아래 부분 페이지당 데이터 수, 페이지 선택
    bottom_menu = window.columns((1, 1, 1))
    with bottom_menu[2]:
        batch_size = st.selectbox("Page Size", options=[1, 9, 16], key=[type,window,3])
    with bottom_menu[1]:
        total_pages = (
            int(len(img) / batch_size) if int(len(img) / batch_size) > 0 else 1
        )
        current_page = st.number_input(
            "Page", min_value=1, max_value=total_pages, step=1
        )
    with bottom_menu[0]:
        st.markdown(f"Page **{current_page}** of **{total_pages}** ")
    pages = split_frame(img, batch_size)
    
    
    
    if 'annotation_num' in pages[0].columns:
        con1.dataframe(data=pages[current_page - 1][['file_name','annotation_num']], use_container_width=True)
    else:
        con1.dataframe(data=pages[current_page - 1]['file_name'], use_container_width=True)
    if not anno.empty:
        show_images(type, pages[current_page - 1][['file_name','id']], anno, con2)
    else:
        show_images(type, pages[current_page - 1][['file_name','id']], pd.DataFrame(), con2)

    if not anno1.empty:
        show_images(type, pages[current_page - 1][['file_name','id']], anno1, con3)
    else:
        show_images(type, pages[current_page - 1][['file_name','id']], pd.DataFrame(), con3)

def main():
    # 원본데이터 확인 가능 아웃풋도 확인하도록 할 수 있을 듯?
    option = st.sidebar.selectbox("데이터 선택",("이미지 데이터", "backup"))
    

    # 데이터 로드
    testd, vald, traind, testjson, valjson, trainjson = load_json_data()
    vald_ = vald.copy()

    if option == "이미지 데이터":
        with st.sidebar.expander("Annotation 선택"):
            st.session_state['Choosed_annotation'] = []
            for category in range(len(categories)):
                if st.checkbox(categories[category],value=True):
                    st.session_state['Choosed_annotation'].append(category)
        # 트레인 데이터 출력
        choose_data = st.sidebar.selectbox("트레인/테스트", ("train","valid","test", 'FNFP'))

        if choose_data == "train":
            st.header("트레인 데이터")

            text = ''
            for idx, c in enumerate(colors2):
                text += f'<span style="color:{c};background:gray;">{categories[idx]} </span>'
            st.markdown(f'<p>{text}</p>', unsafe_allow_html=True)
            traind['annotations'] = traind['annotations'][traind['annotations']['category_id'].isin(st.session_state['Choosed_annotation'])]
            show_dataframe(traind['images'],traind['annotations'],st,'../dataset/')




         ###########################################################
        elif choose_data == "valid":
            st.header("valid 데이터")
            text = ''
            st.markdown(f'<p>{text}</p>', unsafe_allow_html=True)
            
            
            vald['annotations'] = vald['annotations'][vald['annotations']['category_id'].isin(st.session_state['Choosed_annotation'])]
            



            dir = 'valid'
            csv = csv_list(dir)

  
            choose_csv = st.sidebar.selectbox("valid.csv적용",("안함",)+tuple(csv))
            annotationdf = pd.DataFrame()
            
            if choose_csv != "안함":
                st.session_state['confidence'] = st.sidebar.slider("Confidence 값 설정", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
                if st.sidebar.button("현재 csv 백업 폴더로 이동"):
                    csv_to_backup(choose_csv)
                annotationdf = csv_to_dataframe(dir, choose_csv)
                
                ann_num = annotationdf['image_id'].value_counts()

                # vald_['images'] 데이터프레임에 'annotation_num' 열을 추가합니다.
                vald_['images']['annotation_num'] = 0
                for image_id, count in zip(ann_num.index, ann_num.values):
                    vald_['images'].loc[vald_['images']['id'] == image_id, 'annotation_num'] = count
                annotationdf = annotationdf[annotationdf['category_id'].isin(st.session_state['Choosed_annotation'])]


            
            show_dataframe2(vald['images'],vald['annotations'],st,'../dataset/', annotationdf)



            if st.sidebar.button("새 csv 파일 업로드"):
                upload_csv(csv)
    

        
        
        ###########################################################
        
        elif choose_data == "test":
            st.header("테스트 데이터")

            dir = 'output'
            csv = csv_list(dir)

            choose_csv = st.sidebar.selectbox("output.csv적용",("안함",)+tuple(csv))
            annotationdf = pd.DataFrame()
            if choose_csv != "안함":
                st.session_state['confidence'] = st.sidebar.slider("Confidence 값 설정", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
                if st.sidebar.button("현재 csv 백업 폴더로 이동"):
                    csv_to_backup(choose_csv)
                annotationdf = csv_to_dataframe(dir, choose_csv)
                testd['images']['annotation_num'] = annotationdf['image_id'].value_counts()
                annotationdf = annotationdf[annotationdf['category_id'].isin(st.session_state['Choosed_annotation'])]
            
            show_dataframe(testd['images'],annotationdf,st,'../dataset/')

            if st.sidebar.button("새 csv 파일 업로드"):
                upload_csv(csv)

        #############################################################################
        elif choose_data == "FNFP":
            pass

    elif option == "backup":
        if not os.path.exists('./backup/'):
            os.makedirs('./backup/')
        st.header("backup 파일 목록")
        file_list = os.listdir('./backup/')
        for file in file_list:
            file_path = os.path.join('./backup/', file)
            if os.path.isfile(file_path):
                file_name, button1, button2 = st.columns([5,1,2])
                file_name.write(file)
                if button1.button("삭제", key = f"delete {file}"):
                    try:
                        os.remove(file_path)
                        st.success(f"{file} 파일이 삭제되었습니다.")
                    except:
                        st.error("파일 삭제 중 오류가 발생했습니다.")
                    st.rerun()
                if button2.button("기존 폴더로 이동", key = f"move {file}"):
                    try:
                        os.rename(file_path,'./output/'+file)
                        st.success(f"{file} 파일이 이동되었습니다.")
                    except:
                        st.error("파일 이동 중 오류가 발생했습니다.")
                    st.rerun()


if __name__ == "__main__":
    main()