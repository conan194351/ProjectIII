import pickle
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt

# Đọc mô hình từ tệp "model.sav"
with open('model.sav', 'rb') as model_file:
    model = pickle.load(model_file)
Categories = ['cats', 'dogs']
# Đường dẫn tới ảnh mới bạn muốn dự đoán nhãn
path = 'test/test_1.jpg'
img = imread(path)

# Chuẩn bị ảnh mới và dự đoán nhãn
img_resize = resize(img, (150, 150, 3))
l = [img_resize.flatten()]
probability = model.predict_proba(l)

# In kết quả
for ind, val in enumerate(Categories):
    print(f'{val} = {probability[0][ind]*100:.2f}%')

predicted_label = model.predict(l)[0]
print("The predicted image is:", Categories[predicted_label])
plt.imshow(img)
plt.show()