import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

# Генерируем синтетические данные для примера
np.random.seed(42)
n_samples = 1000

# Создаем признаки
age = np.random.normal(35, 10, n_samples)
income = np.random.normal(50000, 20000, n_samples)
credit_history = np.random.uniform(300, 850, n_samples)
debt_ratio = np.random.uniform(0, 1, n_samples)

# Создаем целевую переменную (0 - хороший клиент, 1 - плохой клиент)
# Используем некоторую логику для определения дефолта
probability = 1 / (1 + np.exp(-(
    -0.02 * (age - 35) +
    -0.03 * ((income - 50000) / 10000) +
    -0.02 * ((credit_history - 600) / 100) +
    2 * (debt_ratio - 0.5)
)))
default = (np.random.random(n_samples) < probability).astype(int)

# Создаем DataFrame
data = pd.DataFrame({
    'age': age,
    'income': income,
    'credit_history': credit_history,
    'debt_ratio': debt_ratio,
    'default': default
})

# Разделяем данные на признаки и целевую переменную
X = data.drop('default', axis=1)
y = data['default']

# Разделяем на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Создаем и обучаем модель
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Делаем предсказания
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Оцениваем качество модели
print("Отчет по классификации:")
print(classification_report(y_test, y_pred))
print("\nROC-AUC score:", roc_auc_score(y_test, y_pred_proba))

# Выводим важность признаков
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': abs(model.coef_[0])
})
print("\nВажность признаков:")
print(feature_importance.sort_values('importance', ascending=False))

def predict_credit_risk(age, income, credit_history, debt_ratio):
    """
    Функция для предсказания риска дефолта для нового клиента
    """
    features = np.array([[age, income, credit_history, debt_ratio]])
    features_scaled = scaler.transform(features)
    probability = model.predict_proba(features_scaled)[0][1]
    return probability

# Пример использования
new_client = predict_credit_risk(
    age=30,
    income=45000,
    credit_history=700,
    debt_ratio=0.3
)
print(f"\nВероятность дефолта для нового клиента: {new_client:.2%}")

