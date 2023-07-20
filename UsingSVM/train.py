from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Các vector đặc trưng (mỗi vector là một mẫu)
features = [...]  # Danh sách các vector đặc trưng
labels = [...]  # Danh sách các nhãn tương ứng với vector đặc trưng

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Huấn luyện mô hình SVM
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Đánh giá mô hình trên tập kiểm tra
accuracy = svm_model.score(X_test, y_test)
print("Accuracy:", accuracy)
