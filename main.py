import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.datasets import fetch_openml

# Tải tập dữ liệu
data = fetch_openml('CIFAR_10_small')

X, y = data['data'], data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Hàm đo thời gian và đánh giá hiệu năng
def evaluate_model(model, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    return train_time, predict_time, accuracy, precision, recall


# Khởi tạo mô hình
svm_model = SVC()
knn_model = KNeighborsClassifier(n_neighbors=3)
tree_model = DecisionTreeClassifier()

# Đánh giá từng mô hình
svm_metrics = evaluate_model(svm_model, X_train, X_test, y_train, y_test)
knn_metrics = evaluate_model(knn_model, X_train, X_test, y_train, y_test)
tree_metrics = evaluate_model(tree_model, X_train, X_test, y_train, y_test)

# In kết quả
print(
    "Model: SVM - Training Time: {:.2f}s, Prediction Time: {:.2f}s, Accuracy: {:.2f}, Precision: {:.2f}, Recall: {:.2f}".format(
        *svm_metrics))
print(
    "Model: KNN - Training Time: {:.2f}s, Prediction Time: {:.2f}s, Accuracy: {:.2f}, Precision: {:.2f}, Recall: {:.2f}".format(
        *knn_metrics))
print(
    "Model: Decision Tree - Training Time: {:.2f}s, Prediction Time: {:.2f}s, Accuracy: {:.2f}, Precision: {:.2f}, Recall: {:.2f}".format(
        *tree_metrics))
