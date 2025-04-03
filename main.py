from typing import Dict, Any
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np

import base
from yolo import YOLODetector, yolo
from ocr import OcrDetector
import mixer
from utils import *
from qwen import QwenDetector

#----------------------------------------------------------------------------------------------
# создаем детекторы
#----------------------------------------------------------------------------------------------
CTGR_NAME = ["28.02 Кредит/Ипотека"]
MIXER_NAME = 'mix_28_02_qwen'

class Kat_28_02(YOLODetector):
    def __init__(self):
        super().__init__(
            name="28.02 Кредит/Ипотека", 
            stopvec=np.array([1]),
            names=['credit_or_mortage'],
            detectors = ['yolo_credit'])


class MyOcrDetector_28_02(OcrDetector):
    def __init__(self):
        super().__init__(
            name=CTGR_NAME[0], 
            viptexts=[
                "Ипотека", "кредит", "господдержка", 
                "ипотека предоставляется", "кредит предоставляется", 
                "жилищный кредит", "первоначальный взнос", 
                "процентная ставка", "с господдержкой", 
                "военная ипотека", "субсидированная ипотека", 
                "ЗАЙМЫ ВЫДАЮТСЯ", "ДОГОВОРА ЗАЛОГА"
            ], 
            texts=[
                "в ипотеку", "в кредит", "ипотека для", 
                "Срок кредита", "застройщик берёт", 
                "на себя платежи", "БЕЗ ОТКРЫТИЯ БАНКОВСКОГО", 
                "В ВЫДАЧЕ ЗАЙМА", "Платежи по ипотеке"
            ])

class QwenDetector_28_02(QwenDetector):
    def __init__(self):
        keywords_qwen = [
            # ["Реклама строительства жилья", 
            #               "Реклама застройщика", 
            #               'Реклама специализированных строительных услуг',
            #               'Проектная декларация',
            #               'Реклама жилищного строительства',
            #               'Информация о строительной компании',
            #               'Реклама архитектурных услуг',
            #               'Застройщик',
            #               'Реклама инвестиционно-строительной компании',
            #               'Реклама специализированных застройщиков'
            #               ],
                         [r'\bипотек\w*\b',
                          r'\bкредит\w*\b',
                          ]
                         ]
        qwen_results_file = "/home/tbdbj/forest_test/qwen/qween_ds_1.csv"
        super().__init__(name=CTGR_NAME[0], keywords=keywords_qwen, qwen_file=qwen_results_file)

# # Негативные детекторы
# class Kat_28_08_neg(YOLODetector):
#     def __init__(self):
#         super().__init__(
#             name='28.07 Строительство (ДДУ)', 
#             stopvec=np.array([0]), 
#             names=['real_estate'],
#             detectors = ['yolo_real_estate'] 
#         )
        
# # текстовые детекторы 
# class MyOcrDetector_28_08_neg(OcrDetector):
#     def __init__(self):
#         super().__init__(
#             name='28.07 Строительство (ДДУ)', 
#             viptexts=[
#             ], 
#             texts=[
#                 "жилой квартал", 
#                 "жилой комплекс",
#                 "жилой район",
#                 "долевое участие",
#                 "долевое строительство"
#                 "квартиры",
#                 "проектная декларация",
#             ])
test_ctgr = base.Category(
    name=CTGR_NAME[0], 
    #  detectors=[Kat_28_08(), MyOcrDetector_28_08(), QwenDetector_28_08(), Kat_28_08_neg(), MyOcrDetector_28_08_neg()],
    detectors=[Kat_28_02(), MyOcrDetector_28_02(), QwenDetector_28_02()],
    mixer=mixer.RFRScikit(MIXER_NAME)
)
#----------------------------------------------------------------------------------------------
# тест категории
#----------------------------------------------------------------------------------------------

load_diff_files()

df_res = pd.DataFrame()

def ds_russ2024y_russ2500():
    combined = ds_combine(
        ds_russ2024y,
        ds_russ2500
    )
    for row in combined:
        yield row

for file_name, y, file_path in ds_check_ctgr(CTGR_NAME, ds_russ2024y_russ2500):
    try:
        current_file_name = os.path.basename(file_path)

        local = calc_local_memory(file_path)
        
        if local == None:
            continue
        print('Обработка изображения: ', file_name)

        vec = test_ctgr.calc_vec(local)

        pred = test_ctgr.predict(local)

        row = pd.DataFrame([{
            'file_name': file_name,  # Оригинальное имя файла
            'current_file_name': current_file_name,  # Текущее имя файла
            'category_present': y,
            'threshold_passed': pred,
            'detection_vector': vec,
            'image_text': local['txt']
        }])
        
        df_res = pd.concat([df_res, row], ignore_index=True)
    except Exception as e:
        print('!!!', e)
        continue
  
CSV_PATH = f'mari_qwen/{safe_filename(CTGR_NAME)}_qwen.csv'

os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

df_res['detection_vector'] = df_res['detection_vector'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
df_res.to_csv(CSV_PATH,  index=False, encoding='utf-8')

df = load_df_from_csv(CSV_PATH)

metrix = print_metrix(df=df)
report(test_ctgr=test_ctgr, metrics=metrix, importance=None, dataset='ds_russ2024y+ds_russ2500', suf='_28_08_ds')
#----------------------------------------------------------------------------------------------
# тест обучения
#----------------------------------------------------------------------------------------------
X, Y = load_X_Y_from_csv(test_ctgr, df)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42, stratify=Y)

test_ctgr.mixer.fit(X_train, Y_train, property={
    'n_estimators': 100, 
    'max_depth': 50,
    'min_samples_split': 10,
    'min_samples_leaf': 5,  
    'random_state': 42    
})

test_ctgr.mixer.save(MIXER_NAME)

Y_pred = test_ctgr.mixer.predict(X_test)

metrix = print_metrix(Y_t=Y_test, Y_p=Y_pred)

feature_importances = feature_importances_calc(test_ctgr.mixer, X_train)

report(test_ctgr=test_ctgr, metrics=metrix, importance=feature_importances, dataset='ds_russ2024y+ds_russ2500', suf='_28_08_forest')
