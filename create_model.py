import tensorflow as tf
import csv
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Conv2D, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

# Data Preparation
MODEL_H5 = 'model.h5'

# This is a dummy training data with only 3 observations
# just to see if the model is usable.
# LOGFILE_DIR = 'windows_sim/training_data/0_overfit'

# This is a 2 laps data of driving normally:
# LOGFILE_DIR = 'windows_sim/training_data/1_iterative'

# This data corrects some issues with going off the track.
# LOGFILE_DIR = 'windows_sim/training_data/2_recovery_lap'

# Training data to help the car not to avoid shadows.
# LOGFILE_DIR = 'windows_sim/training_data/3_shadows'

# Additional training data in reverse lap to reduce overfitting further.
# LOGFILE_DIR = 'windows_sim/training_data/4_reverse'

LOGFILE_DIR = 'windows_sim/training_data/reverse'
lines = []
log_dir = os.path.abspath(LOGFILE_DIR)
with open(os.path.join(log_dir, 'driving_log.csv')) as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=32):
  num_samples = len(samples)
  while 1:
    shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]

      images = []
      measurements = []
      for batch_sample in batch_samples:
        corrections = [0, 0.2, -0.2] # center, left, right
        # The following loop takes data from three cameras: center, left, and right.
        # The steering measurement for each camera is then added by
        # the correction as listed above.
        for i, c in enumerate(corrections):
          source_path = batch_sample[i]
          filename = source_path.split('/')[-1]
          current_path = os.path.join(log_dir, 'IMG', os.path.basename(filename))

          image = cv2.imread(current_path)
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          image = np.asarray(image)

          images.append(image)
          measurement = float(batch_sample[3]) + c
          measurements.append(measurement)

          # Flip
          image_flipped = np.fliplr(image)
          images.append(image_flipped)
          measurement_flipped = -measurement
          measurements.append(measurement_flipped)

      X_train = np.array(images)
      y_train = np.array(measurements)
      yield shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

if os.path.exists(MODEL_H5):
  model = load_model(MODEL_H5)
else:
  # Model building
  model = Sequential()
  # YUV Normalization
  # model.add(Lambda(
  #   lambda x: ((x - 16) / (np.matrix([235.0, 240.0, 240.0]) - 16)) - 0.5,
  #   input_shape=(160, 320, 3)))
  # RGB Normalization
  model.add(Lambda(
    lambda x: (x / 255) - 0.5,
    input_shape=(160, 320, 3)))
  model.add(Cropping2D(cropping=((70, 25), (0, 0))))
  # Dropout setup reference:
  # http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
  # Page 1938:
  # Dropout was applied to all the layers of the network with the probability of
  # retaining a hidden unit being p = (0.9, 0.75, 0.75, 0.5, 0.5, 0.5) for the 
  # different layers of the network (going from input to convolutional layers to 
  # fully connected layers).
  model.add(Dropout(0.1))
  model.add(Conv2D(24, (5, 5), strides=(2,2), activation='relu'))
  model.add(Dropout(0.25))
  model.add(Conv2D(36, (5, 5), strides=(2,2), activation='relu'))
  model.add(Dropout(0.25))
  model.add(Conv2D(48, (5, 5), strides=(2,2), activation='relu'))
  model.add(Dropout(0.5))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(Dropout(0.5))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense(100))
  model.add(Dense(50))
  model.add(Dense(10))
  model.add(Dense(1))

optimizer = Adam()
model.compile(loss='mse', optimizer=optimizer)
history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples),
  validation_data=validation_generator, validation_steps=len(validation_samples),
  epochs=5)
model.save(MODEL_H5)

# Plotting
print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()