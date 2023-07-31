import matplotlib
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout
matplotlib.use('Agg')
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Activation
from skimage.io import imread
from skimage.transform import resize
from keras.layers.normalization import BatchNormalization

class DefineModels:
    def __init__(self):
        self.drop = 0.1
        self.freq_axis = 1
        self.channel_axis = 3
        self.filter1 = 64
        self.filter2 = 128
        self.filter3 = 128
        self.filter4 = 128
        self.filter5 = 64

    def modelCNN_8575 (self,in_shape,output):
        name_model = "modelCNN_8575"
        model = Sequential()
        # Input block
        model.add(BatchNormalization(axis=self.freq_axis, name='bn_0_freq', input_shape=in_shape))
        # Conv block 1
        model.add(Conv2D(self.filter1, (3, 3), padding="same", name="conv1"))
        model.add(BatchNormalization(axis=self.channel_axis, name='bn1'))
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))
        model.add(Dropout(self.drop))

        # Conv block 2
        model.add(Conv2D(self.filter2, (3, 3), padding="same", name="conv2"))
        model.add(BatchNormalization(axis=self.channel_axis, name='bn2'))
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))
        model.add(Dropout(self.drop))

        # Conv block 3
        model.add(Conv2D(self.filter3, (3, 3), padding="same", name="conv3"))
        model.add(BatchNormalization(axis=self.channel_axis, name='bn3'))
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='pool3'))
        model.add(Dropout(self.drop))

        # Conv block 4
        model.add(Conv2D(self.filter4, (3, 3), padding="same", name="conv4"))
        model.add(BatchNormalization(axis=self.channel_axis, name='bn4'))
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(3, 3), name='pool4'))
        model.add(Dropout(self.drop))

        # Conv block 5
        model.add(Conv2D(self.filter5, (3, 3), padding="same", name="conv5"))
        model.add(BatchNormalization(axis=self.channel_axis, name='bn5'))
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(2, 2), name='pool5')) # (4,4)
        model.add(Dropout(self.drop))

        # Output
        model.add(Flatten())
        model.add(Dense(units=output, activation='softmax'))

        return model, name_model

# Khởi tạo mô hình
in_shape = (50, 50, 3)  # Thay thế bằng kích thước đầu vào của hình ảnh của bạn
output_classes = 10  # Thay thế bằng số lớp đầu ra của mô hình của bạn
model_builder = DefineModels()
model, model_name = model_builder.modelCNN_8575(in_shape, output_classes)

# Tải trọng số đã lưu vào mô hình
model.load_weights('CRV2/resultCRV2_975/weights.h5')

# Chuẩn bị dữ liệu kiểm tra
test_image_path = 'training_set/dogs/dog.4001.jpg'  # Thay thế bằng đường dẫn đến hình ảnh kiểm tra của bạn
test_image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(50, 50))
test_image_array = tf.keras.preprocessing.image.img_to_array(test_image)
test_image_array = test_image_array.reshape((1,) + test_image_array.shape)  # Mở rộng kích thước batch size thành 1

# Kiểm tra hình ảnh
predictions = model.predict(test_image_array)
predicted_class = tf.argmax(predictions[0], axis=-1)  # Lấy chỉ mục có giá trị dự đoán cao nhất

# In kết quả dự đoán
print("Predicted class:", predicted_class)
