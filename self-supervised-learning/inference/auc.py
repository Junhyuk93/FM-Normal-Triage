import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, roc_curve
import numpy as np

# Load the data from the CSV file
df = pd.read_csv('/workspace/inference/inference.csv') #inference.csv

# Extract the actual labels and predicted probabilities for porosis (class 1)
y_true = df['타겟 클래스']  # 실제 타겟 값
y_prob = df['porosis 확률']  # porosis의 확률 

# 1. original default (threshold=0.5)
# y_pred = df['예측 클래스']  # 예측된 클래스

# 2. threshold custom 하게 설정
threshold =  0.8 # 원하는 threshold 값으로 변경
# Apply the threshold to compute the predicted classes
y_pred = (y_prob >= threshold).astype(int)

# 3. Youden's Index
# ## Calculate the ROC curve
# fpr, tpr, thresholds = roc_curve(y_true, y_prob)
# ## Compute Youden's J Index for each threshold
# youden_index = tpr - fpr
# optimal_idx = np.argmax(youden_index)  # Index of the maximum Youden's J
# threshold = thresholds[optimal_idx]  # Corresponding threshold
# ## Print the optimal threshold
# # print(f"Optimal Threshold (Youden's J): {threshold:.4f}")
# print(threshold)
# ## Apply the optimal threshold to compute the predicted classes
# y_pred = (y_prob >= threshold).astype(int)

# Calculate the AUC
auc = roc_auc_score(y_true, y_prob)

# Calculate the Accuracy
accuracy = accuracy_score(y_true, y_pred)

# Calculate the F1 Score
f1 = f1_score(y_true, y_pred)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
# print('Confusion Matrix:')
# print(cm)

# Confusion Matrix로 Sensitivity (TPR) 및 Specificity (TNR) 계산
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)  # TPR
specificity = tn / (tn + fp)  # TNR

# # 결과 출력
# print(f"AUC: {auc:.4f}")
# print(f"Accuracy: {accuracy:.4f}")
# print(f"F1 Score: {f1:.4f}")
# print(f"Sensitivity (TPR): {sensitivity:.4f}")
# print(f"Specificity (TNR): {specificity:.4f}")

print(pd.DataFrame([threshold,auc,tp,fn,fp,tn,sensitivity,specificity,accuracy,f1]))