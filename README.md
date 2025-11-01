# AI-Agricultural-Spoilage-Risk-Prediction
AI проект по моделированию риска порчи сельскохозяйственной продукции в supply-chain логистике на основе датасета EuroCrop. В проекте построены Gradient Boosting Models и финальная LSTM модель для прогнозирования Spoilage Risk.

## Features
- очистка, нормализация, feature engineering
- RobustScaling / MinMaxScaling
- классы риска High / Medium / Low
- XGBoost / RandomForest baseline models
- primary модель: LSTM (TensorFlow)

## Dataset
EuroCrop_agricultural_logistics_dataset.csv

## Results
LSTM дала наилучшую итоговую метрику по классификации риска порчи (Accuracy / F1 выше baseline GBM моделей).

## File
`main.py` — полный код проекта от загрузки данных до финальной модели + метрики

## Requirements
numpy  
pandas  
scikit-learn  
xgboost  
tensorflow  
matplotlib  
seaborn
