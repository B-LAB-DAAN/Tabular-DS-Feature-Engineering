# Моделирование риска порчи в аграрной логистике с помощью AI (EuroCrop)

Проект посвящён построению AI системы прогнозирования Spoilage Risk для сельскохозяйственных поставок на основе табличных данных EuroCrop. В работе проведена реальная индустриальная подготовка данных, построение feature engineering пайплайна и обучение нескольких ML моделей: XGBoost, Random Forest и финальная модель LSTM, работающая с временной динамикой признаков.

## Цель проекта
Создать модель, способную заранее оценивать уровень риска порчи продукции при аграрной логистике, что имеет прямой бизнес эффект (снижение списаний, оптимизация маршрутов, оптимизация складов, снижение финансовых потерь).

## Что сделано

- проведена обработка данных, очистка, удаление шумовых/бессмысленных признаков
- построены новые фичи (feature interaction, ratio фичи, категориальные энкодинги)
- реализован нормализованный и воспроизводимый табличный pipeline (industry стандарт)
- обучены baseline модели (XGBoost, RandomForest) на регрессию + перевод в risk-категории (High / Medium / Low)
- построена sequence модель LSTM (как финальный best model) на основе временных зависимостей признаков

## Архитектура эксперимента

| Модель | Тип | Назначение |
|-------|-----|-------------|
| XGBoost | gradient boosting baseline | классический ML сильный baseline |
| RandomForest | ensemble baseline | контроль generalization / variance |
| LSTM | deep learning time-sequence model | финальная production модель |

## Dataset
EuroCrop_agricultural_logistics_dataset.csv

## Результаты
LSTM модель показала лучшую итоговую метрику по Accuracy / F1 относительно всех baseline моделей (XGBoost / RF), демонстрируя лучшую устойчивость в прогнозировании реального Spoilage Risk.

## Файл проекта
`Tabular-DS-Feature-Engineering.ipynb` — полный код проекта (EDA → feature engineering → model training → evaluation)

## Requirements
numpy  
pandas  
scikit-learn  
xgboost  
tensorflow  
matplotlib  
seaborn
