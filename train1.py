import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os

# ================================
# 1. SETUP PATHS & CONFIG
# ================================
dataset_path = "C:/Users/benit/OneDrive/Desktop/ml internship/hand_gesture_recognition/leapGestRecog"
img_size = (128, 128)  # MobileNet prefers larger images
batch_size = 32

# ================================
# 2. DATA AUGMENTATION (MobileNet specific)
# ================================
# We use the built-in preprocess_input instead of 1./255
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, 
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = val_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# PRINT THIS AND SAVE IT - Use this order for your Streamlit labels!
print("\n--- EXACT CLASS INDICES (Use this order in Streamlit) ---")
print(train_data.class_indices)
print("----------------------------------------------------------\n")

# ================================
# ================================
# 3. BUILD & FINE-TUNE MOBILENETV2
# ================================
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Unfreeze the last 20 layers of the base model
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5), # High dropout to fight overfitting
    Dense(train_data.num_classes, activation='softmax')
])

# Use a SLIGHTLY lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# ================================
# 4. CALLBACKS & TRAINING
# ================================
checkpoint = ModelCheckpoint("best_gesture_model_mobile.keras", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20, # Transfer learning usually converges faster
    callbacks=[checkpoint, early_stop, reduce_lr]
)

print("Training Complete! Model saved as best_gesture_model_mobile.keras")