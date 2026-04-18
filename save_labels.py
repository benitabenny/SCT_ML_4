from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

dataset_path = "C:/Users/benit/OneDrive/Desktop/ml internship/hand_gesture_recognition/leapGestRecog"

datagen = ImageDataGenerator(validation_split=0.2)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

with open("labels.json", "w") as f:
    json.dump(train_data.class_indices, f)

print("Labels saved:", train_data.class_indices)