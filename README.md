В данном коде используется механизм смешивания (mixer), который в данном контексте реализован с помощью метода Random Forest Regressor или Random Forest Classifier (в зависимости от конкретной реализации в классе mixer.RFRScikit). Это может означать, что деревья решений действительно играют важную роль в этой части кода. Давайте разберем детали:

1. Использование деревьев решений
Случайный лес (Random Forest) — это ансамблевый метод, который использует множество деревьев решений для улучшения точности и снижения риска переобучения. Он обучается путем создания многих деревьев решений на случайно выбранных подмножествах данных и усредняет их результаты (для задач регрессии) или использует голосование (для задач классификации).

2. Конкретная реализация в коде
   test_ctgr.mixer = mixer.RFRScikit(MIXER_NAME)

Здесь создается объект mixer, который, как предполагается, является реализацией случайного леса. Затем происходит его обучение на обучающей выборке:

CopyReplit
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42, stratify=Y)

test_ctgr.mixer.fit(X_train, Y_train, property={
    'n_estimators': 100, 
    'max_depth': 50,
    'min_samples_split': 10,
    'min_samples_leaf': 5,  
    'random_state': 42    
})

3. Параметры случайного леса
n_estimators: Это число деревьев, которые будут созданы в модели случайного леса (в данном случае — 100 деревьев).
max_depth: Максимальная глубина каждого дерева. Ограничение глубины позволяет избежать переобучения.
min_samples_split: Минимальное количество образцов, необходимое для разделения узла. Большее значение может помочь уменьшить переобучение.
min_samples_leaf: Минимальное количество образцов, которые должны быть в листовом узле. Это также помогает контролировать сложность модели.
random_state: Параметр для установки начального состояния генератора случайных чисел, что обеспечивает воспроизводимость результатов.

1. Использование Qwen
Класс QwenDetector_28_02 в коде реализует использование модели Qwen для решения задач, связанных с текстовой аналитикой. Это может включать обработку текстов, извлечение информации из них и обнаружение ключевых терминов, связанных с ипотекой и кредитами.

class QwenDetector_28_02(QwenDetector):
    def __init__(self):
        keywords_qwen = [
            [r'\bипотек\w*\b',
             r'\bкредит\w*\b',
            ]
        ]
        qwen_results_file = "/home/tbdbj/forest_test/qwen/qween_ds_1.csv"
        super().__init__(name=CTGR_NAME[0], keywords=keywords_qwen, qwen_file=qwen_results_file)
  keywords_qwen: В этом списке определены регулярные выражения, которые ищут слова, связанные с «ипотекой» и «кредитом». Это позволяет детектору Qwen находить в изображениях или текстах упоминания этих терминов.

qwen_file: qwen_results_file указывает путь к CSV-файлу, в котором, вероятно, будут сохраняться результаты работы Qwen.

2. Использование OCR
Класс MyOcrDetector_28_02 применяется для детекции текста на изображениях с использованием метода OCR.
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
            ]
        )
   viptexts: Содержит список ключевых слов и фраз, на которые следует обратить особое внимание. Это помогает выделять важные тексты, связанные с ипотеками и кредитами.

texts: Содержит более общие фразы, которые также могут встретиться в документах или изображениях, и их необходимо распознавать.

3. Процесс обработки
В основном цикле обработки изображений (в функции обработки категорий):
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
            'file_name': file_name,
            'current_file_name': current_file_name,
            'category_present': y,
            'threshold_passed': pred,
            'detection_vector': vec,
            'image_text': local['txt']
        }])
   Здесь происходит выполнение методов, связанных как с Qwen, так и с OCR. local содержит результаты обработки, полученные путем применения методов OCR и Qwen. 

vec = test_ctgr.calc_vec(local) и pred = test_ctgr.predict(local) используют разные детекторы (включая Qwen и OCR) для получения предсказаний и векторов детекций.

   В коде используются оба метода: Qwen для обработки и анализа текстов с помощью языковой модели и OCR для распознавания текста на изображениях,в коде использовались деревья, а именно деревья решений, реализованные в методе случайного леса.
# OCR_YOLO_Qwen_derev
