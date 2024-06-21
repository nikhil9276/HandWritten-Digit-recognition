import numpy as np
import matplotlib.pyplot as plt

import keras

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout
(x_train,y_train),(x_test,y_test)=mnist.load_data()
def plot_input_img(i):
    plt.imshow(x_train[i],cmap='binary')
    plt.title(y_train[i])
    plt.show()
 # pre process the images
# Normalizing the image to [0,1] range
x_train=x_train.astype(np.float32)/255
x_test=x_test.astype(np.float32)/255

# reshape the dimension of images to(28,28,1)
x_train=np.expand_dims(x_train,-1)
x_test=np.expand_dims(x_test,-1)

# convert classes to one hot vectors

y_train=keras.utils.to_categorical(y_train)
y_test=keras.utils.to_categorical(y_test)
model=Sequential()

model.add(Conv2D(32,(3,3),input_shape=(28,28,1),activation='relu'))
model.add(MaxPool2D((2,2)))


model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D((2,2)))

model.add(Flatten())

model.add(Dropout(0.25))
model.add(Dense(10,activation="softmax"))
model.summary()
" " " output
Model: "sequential_5"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d_3 (Conv2D)               │ (None, 26, 26, 32)     │           320 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 13, 13, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_4 (Conv2D)               │ (None, 11, 11, 64)     │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_3 (MaxPooling2D)  │ (None, 5, 5, 64)       │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 1600)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 1600)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 10)             │        16,010 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 34,826 (136.04 KB)
 Trainable params: 34,826 (136.04 KB)
 Non-trainable params: 0 (0.00 B)  
                                  " " "
model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Early stopping
es = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=4, verbose=1)

# Model checkpoint
mc = ModelCheckpoint("./bestmodel.keras", monitor="val_accuracy", verbose=1, save_best_only=True)

cb = [es, mc]
#model training


his=model.fit(x_train,y_train,epochs=50,validation_split=0.3,callbacks=cb)

   """
Epoch 1/50
1308/1313 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.9981 - loss: 0.0051
Epoch 1: val_accuracy improved from -inf to 0.99083, saving model to ./bestmodel.keras
1313/1313 ━━━━━━━━━━━━━━━━━━━━ 7s 5ms/step - accuracy: 0.9981 - loss: 0.0051 - val_accuracy: 0.9908 - val_loss: 0.0582
Epoch 2/50
1310/1313 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.9985 - loss: 0.0051
Epoch 2: val_accuracy did not improve from 0.99083
1313/1313 ━━━━━━━━━━━━━━━━━━━━ 8s 6ms/step - accuracy: 0.9985 - loss: 0.0051 - val_accuracy: 0.9899 - val_loss: 0.0619
Epoch 3/50
1309/1313 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.9984 - loss: 0.0047
Epoch 3: val_accuracy improved from 0.99083 to 0.99094, saving model to ./bestmodel.keras
1313/1313 ━━━━━━━━━━━━━━━━━━━━ 8s 6ms/step - accuracy: 0.9984 - loss: 0.0047 - val_accuracy: 0.9909 - val_loss: 0.0609
Epoch 4/50
1310/1313 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.9985 - loss: 0.0045
Epoch 4: val_accuracy did not improve from 0.99094
1313/1313 ━━━━━━━━━━━━━━━━━━━━ 8s 6ms/step - accuracy: 0.9985 - loss: 0.0045 - val_accuracy: 0.9907 - val_loss: 0.0548
Epoch 5/50
1309/1313 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 0.9983 - loss: 0.0051
Epoch 5: val_accuracy improved from 0.99094 to 0.99117, saving model to ./bestmodel.keras
1313/1313 ━━━━━━━━━━━━━━━━━━━━ 8s 6ms/step - accuracy: 0.9983 - loss: 0.0051 - val_accuracy: 0.9912 - val_loss: 0.0548
Epoch 5: early stopping

"""
model.save('my_trained_model.keras')
model_S=keras.models.load_model("my_trained_model.keras")
score= model_S.evaluate(x_test,y_test)
print(f" the model accurecy is {score[1]}")
     """
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9895 - loss: 0.0582
 the model accurecy is 0.9921000003814697
 """
