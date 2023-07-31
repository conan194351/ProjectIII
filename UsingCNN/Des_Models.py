import matplotlib
from keras.models import Sequential
# from keras.layers.recurrent import LSTM
from keras.layers import Dense, Flatten, Dropout

matplotlib.use('Agg')
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Conv2D


from keras.layers.activation import ELU
from keras.layers import BatchNormalization

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
        # model.add(Conv2D(64, (3, 3), padding="same",name="conv1",kernel_initializer='random_normal',
        #                 bias_initializer='zeros')) # 256 0.86667
        model.add(Conv2D(self.filter1, (3, 3), padding="same", name="conv1",
                         # kernel_initializer='random_uniform',
                         # bias_initializer='zeros'
                         ))  # 256 0.86667

        # model.add(Conv2D(128, (3, 3), padding="same", name="conv1", input_shape=input_shape))
        model.add(BatchNormalization(axis=self.channel_axis, name='bn1'))
        model.add(ELU())
        model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))
        model.add(Dropout(self.drop))

        # Conv block 2
        model.add(Conv2D(self.filter2, (3, 3), padding="same", name="conv2",
                         #kernel_initializer='random_uniform',
                         #bias_initializer='zeros'
                         ))
        model.add(BatchNormalization(axis=self.channel_axis, name='bn2'))
        model.add(ELU())
        model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))
        model.add(Dropout(self.drop))

        # Conv block 3
        model.add(Conv2D(self.filter3, (3, 3), padding="same", name="conv3",
                         # kernel_initializer='random_uniform',
                         # bias_initializer='zeros'
                         ))
        model.add(BatchNormalization(axis=self.channel_axis, name='bn3'))
        model.add(ELU())
        model.add(MaxPooling2D(pool_size=(2, 2), name='pool3'))
        model.add(Dropout(self.drop))

        # Conv block 4
        model.add(Conv2D(self.filter4, (3, 3), padding="same", name="conv4",
                         # kernel_initializer='random_uniform',
                         # bias_initializer='zeros'
                         ))
        model.add(BatchNormalization(axis=self.channel_axis, name='bn4'))
        model.add(ELU())
        model.add(MaxPooling2D(pool_size=(3, 3), name='pool4'))
        model.add(Dropout(self.drop))

        # Conv block 5
        model.add(Conv2D(self.filter5, (3, 3), padding="same", name="conv5",
                         # kernel_initializer='random_uniform',
                         # bias_initializer='zeros'
                         ))
        model.add(BatchNormalization(axis=self.channel_axis, name='bn5'))
        model.add(ELU())
        model.add(MaxPooling2D(pool_size=(2, 2), name='pool5')) # (4,4)
        model.add(Dropout(self.drop))
        #print (model.output_shape)

        # Output
        model.add(Flatten())
      
        model.add(Dense(units=output, activation='softmax',
                        # kernel_initializer='random_uniform',
                        # bias_initializer='zeros'
                        ))
        return model, name_model

   