{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "8bc4f801-3b43-4845-82da-13c0ce242ffd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: tensorflow in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (2.16.1)\n",
      "Requirement already satisfied: tensorflow-intel==2.16.1 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from tensorflow) (2.16.1)\n",
      "Requirement already satisfied: absl-py>=1.0.0 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (2.1.0)\n",
      "Requirement already satisfied: astunparse>=1.6.0 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (1.6.3)\n",
      "Requirement already satisfied: flatbuffers>=23.5.26 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (24.3.25)\n",
      "Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (0.5.4)\n",
      "Requirement already satisfied: google-pasta>=0.1.1 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (0.2.0)\n",
      "Requirement already satisfied: h5py>=3.10.0 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (3.11.0)\n",
      "Requirement already satisfied: libclang>=13.0.0 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (18.1.1)\n",
      "Requirement already satisfied: ml-dtypes~=0.3.1 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (0.3.2)\n",
      "Requirement already satisfied: opt-einsum>=2.3.2 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (3.3.0)\n",
      "Requirement already satisfied: packaging in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (23.1)\n",
      "Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (3.20.3)\n",
      "Requirement already satisfied: requests<3,>=2.21.0 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (2.31.0)\n",
      "Requirement already satisfied: setuptools in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (68.2.2)\n",
      "Requirement already satisfied: six>=1.12.0 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (1.16.0)\n",
      "Requirement already satisfied: termcolor>=1.1.0 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (2.4.0)\n",
      "Requirement already satisfied: typing-extensions>=3.6.6 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (4.9.0)\n",
      "Requirement already satisfied: wrapt>=1.11.0 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (1.14.1)\n",
      "Requirement already satisfied: grpcio<2.0,>=1.24.3 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (1.64.1)\n",
      "Requirement already satisfied: tensorboard<2.17,>=2.16 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (2.16.2)\n",
      "Requirement already satisfied: keras>=3.0.0 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (3.3.3)\n",
      "Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (0.31.0)\n",
      "Requirement already satisfied: numpy<2.0.0,>=1.23.5 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.16.1->tensorflow) (1.26.4)\n",
      "Requirement already satisfied: wheel<1.0,>=0.23.0 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from astunparse>=1.6.0->tensorflow-intel==2.16.1->tensorflow) (0.41.2)\n",
      "Requirement already satisfied: rich in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from keras>=3.0.0->tensorflow-intel==2.16.1->tensorflow) (13.3.5)\n",
      "Requirement already satisfied: namex in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from keras>=3.0.0->tensorflow-intel==2.16.1->tensorflow) (0.0.8)\n",
      "Requirement already satisfied: optree in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from keras>=3.0.0->tensorflow-intel==2.16.1->tensorflow) (0.11.0)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorflow-intel==2.16.1->tensorflow) (2.0.4)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorflow-intel==2.16.1->tensorflow) (3.4)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorflow-intel==2.16.1->tensorflow) (2.0.7)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorflow-intel==2.16.1->tensorflow) (2024.2.2)\n",
      "Requirement already satisfied: markdown>=2.6.8 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from tensorboard<2.17,>=2.16->tensorflow-intel==2.16.1->tensorflow) (3.4.1)\n",
      "Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from tensorboard<2.17,>=2.16->tensorflow-intel==2.16.1->tensorflow) (0.7.2)\n",
      "Requirement already satisfied: werkzeug>=1.0.1 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from tensorboard<2.17,>=2.16->tensorflow-intel==2.16.1->tensorflow) (2.2.3)\n",
      "Requirement already satisfied: MarkupSafe>=2.1.1 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from werkzeug>=1.0.1->tensorboard<2.17,>=2.16->tensorflow-intel==2.16.1->tensorflow) (2.1.3)\n",
      "Requirement already satisfied: markdown-it-py<3.0.0,>=2.2.0 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from rich->keras>=3.0.0->tensorflow-intel==2.16.1->tensorflow) (2.2.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from rich->keras>=3.0.0->tensorflow-intel==2.16.1->tensorflow) (2.15.1)\n",
      "Requirement already satisfied: mdurl~=0.1 in c:\\users\\nikhi\\anaconda3\\lib\\site-packages (from markdown-it-py<3.0.0,>=2.2.0->rich->keras>=3.0.0->tensorflow-intel==2.16.1->tensorflow) (0.1.0)\n"
     ]
    }
   ],
   "source": [
    "!pip install tensorflow\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "ce4f6a66-ca1b-4c35-8d9b-fa2541255174",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz\n",
      "\u001b[1m11490434/11490434\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m3s\u001b[0m 0us/step\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\nikhi\\anaconda3\\Lib\\site-packages\\keras\\src\\layers\\convolutional\\base_conv.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
      "  super().__init__(activity_regularizer=activity_regularizer, **kwargs)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/5\n",
      "\u001b[1m1500/1500\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m47s\u001b[0m 28ms/step - accuracy: 0.8975 - loss: 0.3331 - val_accuracy: 0.9855 - val_loss: 0.0512\n",
      "Epoch 2/5\n",
      "\u001b[1m1500/1500\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m41s\u001b[0m 28ms/step - accuracy: 0.9848 - loss: 0.0478 - val_accuracy: 0.9869 - val_loss: 0.0420\n",
      "Epoch 3/5\n",
      "\u001b[1m1500/1500\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m42s\u001b[0m 28ms/step - accuracy: 0.9906 - loss: 0.0289 - val_accuracy: 0.9881 - val_loss: 0.0375\n",
      "Epoch 4/5\n",
      "\u001b[1m1500/1500\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m42s\u001b[0m 28ms/step - accuracy: 0.9936 - loss: 0.0215 - val_accuracy: 0.9869 - val_loss: 0.0459\n",
      "Epoch 5/5\n",
      "\u001b[1m1500/1500\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m42s\u001b[0m 28ms/step - accuracy: 0.9944 - loss: 0.0158 - val_accuracy: 0.9891 - val_loss: 0.0392\n",
      "\u001b[1m313/313\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m4s\u001b[0m 11ms/step - accuracy: 0.9872 - loss: 0.0410\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test accuracy: 0.9907000064849854\n"
     ]
    }
   ],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow.keras.datasets import mnist\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense\n",
    "import numpy as np\n",
    "import joblib\n",
    "\n",
    "# Load the MNIST dataset\n",
    "(x_train, y_train), (x_test, y_test) = mnist.load_data()\n",
    "\n",
    "# Normalize the data\n",
    "x_train, x_test = x_train / 255.0, x_test / 255.0\n",
    "\n",
    "# Reshape the data to fit the model\n",
    "x_train = x_train.reshape(-1, 28, 28, 1)\n",
    "x_test = x_test.reshape(-1, 28, 28, 1)\n",
    "\n",
    "# Build the model\n",
    "model = Sequential([\n",
    "    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),\n",
    "    MaxPooling2D(pool_size=(2, 2)),\n",
    "    Conv2D(64, kernel_size=(3, 3), activation='relu'),\n",
    "    MaxPooling2D(pool_size=(2, 2)),\n",
    "    Flatten(),\n",
    "    Dense(128, activation='relu'),\n",
    "    Dense(10, activation='softmax')\n",
    "])\n",
    "\n",
    "# Compile the model\n",
    "model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n",
    "\n",
    "# Train the model\n",
    "model.fit(x_train, y_train, epochs=5, validation_split=0.2)\n",
    "\n",
    "# Evaluate the model\n",
    "loss, accuracy = model.evaluate(x_test, y_test)\n",
    "print(f'Test accuracy: {accuracy}')\n",
    "\n",
    "# Save the trained model\n",
    "model.save('mnist_cnn_model.h5')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3cb38dec-0e6a-4614-81ec-63649337a30f",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  },
  "widgets": {
   "application/vnd.jupyter.widget-state+json": {
    "state": {},
    "version_major": 2,
    "version_minor": 0
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
