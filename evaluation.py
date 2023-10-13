import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report

def create_test_data_generator(input_shape, batch_size, data_directory):
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    test_generator = test_datagen.flow_from_directory(
        data_directory,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # Set shuffle to False for reproducible evaluation
    )

    return test_generator

if __name__ == "__main__":
    input_shape = (100, 100, 3)
    num_classes = 4
    batch_size = 32
    test_data_directory = r"C:\Users\Namita Behera\Desktop\paddy_data\NitrogenDeficiencyImage\Test"

    model = load_model("nitrogen_deficiency_model.h5")

    # Create the test data generator
    test_generator = create_test_data_generator(input_shape, batch_size, test_data_directory)

    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(test_generator)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)

    # Get the predicted class probabilities for test data
    predictions = model.predict(test_generator)

    # Convert predicted probabilities to predicted class labels
    predicted_classes = predictions.argmax(axis=-1)

    # Get the true labels from the test generator
    true_labels = test_generator.classes

    # Calculate precision, recall, and F1 score
    precision = precision_score(true_labels, predicted_classes, average='weighted')
    recall = recall_score(true_labels, predicted_classes, average='weighted')
    f1 = f1_score(true_labels, predicted_classes, average='weighted')

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # Calculate and print accuracy
    accuracy = accuracy_score(true_labels, predicted_classes)
    print("Accuracy:", accuracy)

    # Get the class labels from the test generator
    class_labels = list(test_generator.class_indices.keys())

    # Generate a classification report
    report = classification_report(true_labels, predicted_classes, target_names=class_labels)
    print(report)
