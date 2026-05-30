import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

os.system("cls")
# =========================
# 1. Cargar datos
# =========================
ruta = r"C:/Users/Mateo Garcia/OneDrive/Escritorio/bank_transactions_data_2.csv"
df = pd.read_csv(ruta)

# =========================
# 2. Variables
# =========================
features = ['TransactionAmount', 'TransactionDuration']
X = df[features]

# =========================
# 3. Limpieza
# =========================
X = X.fillna(X.median())

# =========================
# 4. Escalamiento
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 5. DBSCAN
# =========================
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X_scaled)
df['Fraude'] = (labels == -1).astype(int)

# Resultados DBSCAN
total_fraudes = df['Fraude'].sum()
print(f"Total fraudes detectados (DBSCAN): {total_fraudes}")

print("\nPromedios en fraudes (DBSCAN):")
print(df[df['Fraude'] == 1][features].mean())

# =========================
# 6. Gráfica DBSCAN
# =========================
plt.figure()

# todos los puntos
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.3)

# fraudes
plt.scatter(
    X_scaled[df['Fraude'] == 1, 0],
    X_scaled[df['Fraude'] == 1, 1],
    c='black',
    label='Possible Fraud'
)

plt.title("DBSCAN Fraud Detection")
plt.xlabel("TransactionAmount")
plt.ylabel("TransactionDuration")
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()

# =========================
# 7. Train/Test
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, df['Fraude'], test_size=0.3, random_state=42
)

# =========================
# 8. REGRESIÓN LOGÍSTICA
# =========================
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print("\n--- REGRESIÓN LOGÍSTICA ---")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred_log))
print("Reporte:\n", classification_report(y_test, y_pred_log))

# =========================
# 9. ÁRBOL DE DECISIÓN
# =========================
tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)

print("\n--- ÁRBOL DE DECISIÓN ---")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred_tree))
print("Reporte:\n", classification_report(y_test, y_pred_tree))

# =========================
# 10. Visualizar árbol
# =========================
plt.figure(figsize=(10,6))
plot_tree(
    tree_model,
    feature_names=features,
    class_names=["No Fraud","Fraud"],
    filled=True
)
plt.title("Decision Tree")
plt.show()

# =========================
# 11. Gráficas comparativas (UNA SOLA FIGURA)
# =========================
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# ---- LOGÍSTICA ----
axs[0].scatter(X_test[:, 0], X_test[:, 1], alpha=0.3)

# TP (verde)
tp = (y_test == 1) & (y_pred_log == 1)
axs[0].scatter(X_test[tp, 0], X_test[tp, 1], c='green', label='TP')

# FN (naranja)
fn = (y_test == 1) & (y_pred_log == 0)
axs[0].scatter(X_test[fn, 0], X_test[fn, 1], c='red', label='FN')

# FP (rojo)
fp = (y_test == 0) & (y_pred_log == 1)
axs[0].scatter(X_test[fp, 0], X_test[fp, 1], c='orange', label='FP')

axs[0].set_title("Logistic Regression")
axs[0].set_xlabel("TransactionAmount")
axs[0].set_ylabel("TransactionDuration")
axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1))

# ---- ÁRBOL ----
axs[1].scatter(X_test[:, 0], X_test[:, 1], alpha=0.3)

# TP
tp = (y_test == 1) & (y_pred_tree == 1)
axs[1].scatter(X_test[tp, 0], X_test[tp, 1], c='green', label='TP')

# FN
fn = (y_test == 1) & (y_pred_tree == 0)
axs[1].scatter(X_test[fn, 0], X_test[fn, 1], c='red', label='FN')

# FP
fp = (y_test == 0) & (y_pred_tree == 1)
axs[1].scatter(X_test[fp, 0], X_test[fp, 1], c='orange', label='FP')

axs[1].set_title("Decision Tree")
axs[1].set_xlabel("TransactionAmount")
axs[1].set_ylabel("TransactionDuration")
axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.suptitle("Model Comparison (TP, FP, FN)")
plt.tight_layout()
plt.show()