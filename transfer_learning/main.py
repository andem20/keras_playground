import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.engine import keras_tensor


# Remove currupted images from datasets
dirname = os.path.dirname(__file__) + '/' # Get the absolute path
# num_skipped = 0
# for folder_name in ("Cat", "Dog"):
#     folder_path = os.path.join(dirname + "PetImages", folder_name)
#     for fname in os.listdir(folder_path):
#         fpath = os.path.join(folder_path, fname)
#         try:
#             fobj = open(fpath, "rb")
#             is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
#         finally:
#             fobj.close()

#         if not is_jfif:
#             num_skipped += 1
#             # Delete corrupted image
#             os.remove(fpath)

# print("Deleted %d images" % num_skipped)

# Attributes
image_size = (180, 180)
batch_size = 32
data_dir = 'data'
epochs = 10

# Cats = 0, Dogs = 1

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dirname + data_dir,
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='int'
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dirname + data_dir,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='int'
)

# # Plotting 9 samples of the data
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(2):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(int(labels[i]))
#         plt.axis("off")

# plt.show()

base_model = keras.applications.Xception(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(image_size[0], image_size[1], 3), 
    include_top=False)  # Do not include the ImageNet classifier at the top.


base_model.trainable = False # Freeze the base model

# Creating a new model on top
inputs = keras.Input(shape=(image_size[0], image_size[1], 3))
# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning, as you will
# learn in a few paragraphs.
scale_layer = keras.layers.experimental.preprocessing.Rescaling(scale=1 / 127.5, offset=-1)
x = scale_layer(inputs)
x = base_model(x, training=False)
# Convert features of shape `base_model.output_shape[1:]` to vectors
x = keras.layers.GlobalAveragePooling2D()(x)
# A Dense classifier with a single unit (binary classification)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)

# Training the model
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=[keras.metrics.BinaryAccuracy()])
model.fit(train_ds, epochs=epochs, validation_data=val_ds)


# Predictions
for image in os.listdir(dirname + 'PetImages/elli'):
    cat_img = keras.preprocessing.image.load_img(dirname + 'PetImages/elli/' + image, target_size=image_size)
    cat_img_array = keras.preprocessing.image.img_to_array(cat_img)
    cat_img_array = tf.expand_dims(cat_img_array, 0)
    prediction = model.predict(cat_img_array)
    score = prediction[0]

    print(
        "Image: %s, is %.2f percent cat and %.2f percent dog."
        % (image, 100 * (1 - score), 100 * score)
    )