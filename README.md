# Unsupervised pretraining in computer vision
Experiments with unsupervised pretraining in computer vision.

## Установка

```
git clone https://github.com/ivan-chai/unsupervised-pretraining
cd unsupervised-pretraining/
pip install -e .
```

## Конфигурирование запуска
### Конфигурирование и запуск оценщика

Файл конфига претренированной нейросети должен находиться в директории models/. 
В конфиге в секии model должны быть указаны URL до модели и проверочного эмбеддинга.
В секции augmentations необходимо указать используемые при предобработке аугментации.
В качестве примера см. models/sota_resnet_example.  

Параметры тренера лежат в конфиге trainer/trainer.yaml. По дефолту при отсутствии улучшения валидационного лосса на 
протяжении 25 эпох останавливается тренировка.  
Дефолтные коллбэки тренировки классификатора определны в callbacks/evaluator_callbacks.yaml, при желании можно добавить свой.

Запуск тренировки производится командой:
```
ulib_evaluate '+model="./path/to/model/config"'
```
