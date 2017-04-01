import csv
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split

# Data Preparation
MODEL_H5 = 'model.h5'

LOGFILE_DIR = 'windows_sim/training_data/0_overfit'

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
      yield X_train, y_train

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
train = next(train_generator)

def crop(image):
  return image[70:135,:,:]
  # return image

# cv2.imshow('Center', crop(train[0][0]))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('Left', crop(train[0][2]))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('Right', crop(train[0][4]))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite('c.jpg', train[0][0])
cv2.imwrite('l.jpg', train[0][2])
cv2.imwrite('r.jpg', train[0][4])

cv2.imwrite('c-processed.jpg', crop(train[0][0]))
cv2.imwrite('l-processed.jpg', crop(train[0][2]))
cv2.imwrite('r-processed.jpg', crop(train[0][4]))