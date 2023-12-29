import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, ZeroPadding2D, MaxPooling2D, AveragePooling2D, Flatten, Dense , Rescaling
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

class ResNetBlock:
    def __init__(self, initializer, training=True):
        self.initializer = initializer
        self.training = training

    def identity_block(self, X, f,filters):
        F1, F2, F3 = filters
        X_shortcut = X

        X = Conv2D(F1, (1, 1), strides=(1, 1), padding="valid", kernel_initializer=self.initializer)(X)
        X = BatchNormalization(axis=3)(X, training=self.training)
        X = ReLU()(X)

        X = Conv2D(F2, (f, f), strides=(1, 1), padding="same", kernel_initializer=self.initializer)(X)
        X = BatchNormalization(axis=3)(X, training=self.training)
        X = ReLU()(X)

        X = Conv2D(F3, (1, 1), strides=(1, 1), padding="valid", kernel_initializer=self.initializer)(X)
        X = BatchNormalization(axis=3)(X, training=self.training)

        X = Add()([X, X_shortcut])
        X = ReLU()(X)

        return X

    def convolutional_block(self, X, f,filters, s):
        F1, F2, F3 = filters
        X_shortcut = X

        X = Conv2D(F1, (1, 1), strides=(s, s), padding="valid", kernel_initializer=self.initializer)(X)
        X = BatchNormalization(axis=3)(X, training=self.training)
        X = ReLU()(X)

        X = Conv2D(F2, (f, f), strides=(1, 1), padding="same", kernel_initializer=self.initializer)(X)
        X = BatchNormalization(axis=3)(X, training=self.training)
        X = ReLU()(X)

        X = Conv2D(F3, (1, 1), strides=(1, 1), padding="valid", kernel_initializer=self.initializer)(X)
        X = BatchNormalization(axis=3)(X, training=self.training)

        X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), padding="valid", kernel_initializer=self.initializer)(X_shortcut)
        X_shortcut = BatchNormalization(axis=3)(X_shortcut, training=self.training)

        X = Add()([X, X_shortcut])
        X = ReLU()(X)

        return X


class ResNet50Model:
    def __init__(self, input_shape=(64, 64, 3), classes=3):
        self.input_shape = input_shape
        self.classes = classes

    def build_model(self, training=False):
        X_input = Input(self.input_shape)
        X_input = Rescaling(scale=1./255)(X_input)
        X_input = self._data_augmentation()(X_input)

        X = ZeroPadding2D((3, 3))(X_input)

        # Stage 1
        X = Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3)(X, training=training)
        X = ReLU()(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        # Stages 2-5
        resnet_block = ResNetBlock(tf.keras.initializers.Constant(1), training)
        X = resnet_block.convolutional_block(X, 3,[64, 64, 256] ,1)
        X = resnet_block.identity_block(X, 3,[64, 64, 256])
        X = resnet_block.identity_block(X, 3,[64, 64, 256])

        X = resnet_block.convolutional_block(X, 3, [128,128,512],2)
        X = resnet_block.identity_block(X, 3, [128,128,512])
        X = resnet_block.identity_block(X, 3, [128,128,512])
        X = resnet_block.identity_block(X, 3, [128,128,512])

        X = resnet_block.convolutional_block(X, 3,[256,256,1024] ,2)
        X = resnet_block.identity_block(X, 3,[256,256,1024])
        X = resnet_block.identity_block(X, 3,[256,256,1024])
        X = resnet_block.identity_block(X, 3,[256,256,1024])
        X = resnet_block.identity_block(X, 3,[256,256,1024])
        X = resnet_block.identity_block(X, 3,[256,256,1024])

        X = resnet_block.convolutional_block(X, 3, [512,512,2048],2)
        X = resnet_block.identity_block(X, 3,[512,512,2048])
        X = resnet_block.identity_block(X, 3,[512,512,2048])

        # Final Stage
        X = AveragePooling2D(pool_size=(2, 2))(X)
        X = Flatten()(X)
        X = Dense(self.classes, activation="softmax", kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)

        model = Model(inputs=X_input, outputs=X)

        return model

    def _data_augmentation(self):
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2)
        ])

        return data_augmentation


def train_model(model, train_dataset, validation_dataset, initial_epochs=1):
    opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=opt, loss=loss_function, metrics=["accuracy"])

    checkpoint_path = 'model_checkpoint.h5'
    checkpoint_callback = ModelCheckpoint(checkpoint_path, save_weights_only=True, save_best_only=True)

    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)

    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=initial_epochs, callbacks=[checkpoint_callback])

    return model, history


def main():
    np.random.seed(1)
    tf.random.set_seed(2)

    BATCH_SIZE = 4
    IMG_SIZE = (64, 64)
    # directory = "data/chest_xray/train/"
    # train_dataset = image_dataset_from_directory(directory,
    #                                             shuffle=True,
    #                                             batch_size=BATCH_SIZE,
    #                                             image_size=IMG_SIZE,
    #                                             validation_split=0.2,
    #                                             subset='training',
    #                                             seed=42)
    # validation_dataset = image_dataset_from_directory(directory,
    #                                             shuffle=True,
    #                                             batch_size=BATCH_SIZE,
    #                                             image_size=IMG_SIZE,
    #                                             validation_split=0.2,
    #                                             subset='validation',
    #                                          seed=42)

    # class_names = train_dataset.class_names

    # resnet_model = ResNet50Model(input_shape=(64, 64, 3), classes=len(class_names))
    # model = resnet_model.build_model(training=True)

    # trained_model, training_history = train_model(model, train_dataset, validation_dataset)

    # trained_model.save("path/to/save/model_resnet_pneumonia")

  

    # # Load the SavedModel from the .pb file
    # trained_model = tf.saved_model.load('path/to/save/model_resnet_pnemonia')
    # # print(trained_model.__class__)


    # Load the trained model
    model = tf.keras.models.load_model("path/to/save/model_resnet_pneumonia")

    # Define the class names
    class_names = ["normal", "bacteria_pneumonia", "virus_pneumonia"]  # Replace with your actual class names

    # Load and preprocess an example image for prediction
    img_path = "data/chest_xray/train/pneumonia_virus/person1642_virus_2842.jpeg"
    img = image.load_img(img_path, target_size=(64, 64))  # Adjust the target_size based on your model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values to the range [0, 1]

    # Make predictions
    predictions = model.predict(img_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions[0])

    # Get the predicted class label
    predicted_class_label = class_names[predicted_class_index]

    # Print the predicted class label
    print("Predicted class:", predicted_class_label)

if __name__ == "__main__":
    main()



