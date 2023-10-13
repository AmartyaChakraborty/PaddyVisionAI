import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_resnet_model(input_shape, num_classes):
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def create_data_generators(input_shape, batch_size, data_directory):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        data_directory,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator

def train_model(model, train_generator, epochs, steps_per_epoch):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch)

if __name__ == "__main__":
    input_shape = (100, 100, 3)  # Height, Width, and Channels (RGB)
    num_classes = 4  # Number of different nitrogen deficiency levels (swap1, swap2, swap3, swap4)
    batch_size = 32
    epochs = 10
    steps_per_epoch = 1407 // batch_size  # Number of images in each folder divided by batch size

    data_directory = r"C:\Users\Namita Behera\Desktop\paddy_data\NitrogenDeficiencyImage\Training"

    model = build_resnet_model(input_shape, num_classes)

    train_generator = create_data_generators(input_shape, batch_size, data_directory)

    train_model(model, train_generator, epochs, steps_per_epoch)

    # Save the model if you are satisfied with its performance
    model.save("nitrogen_deficiency_model.h5")
