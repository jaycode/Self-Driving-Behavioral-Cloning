# Behavioral Cloning Project

The goals / steps of this project are the following:

- Use the simulator to collect data of good driving behavior
- Build, a convolution neural network in Keras that predicts steering angles from images
- Train and validate the model with a training and validation set
- Test that the model successfully drives around track one without leaving the road
- Summarize the results with a written report

[//]: # (Image References)

[no_dropout-overfit]: ./doc_imgs/no_dropout-overfit.png "No Dropout - Overfit"
[no_dropout-iterative]: ./doc_imgs/no_dropout-iterative.png "No Dropout - Iterative"
[dropout-overfit]: ./doc_imgs/dropout-overfit.png "Dropout - Overfit"
[dropout-iterative]: ./doc_imgs/dropout-iterative.png "Dropout - Iterative"

[nvidianet]: ./doc_imgs/NVIDIANet.PNG "NVIDIA Net"

[c]: ./doc_imgs/c.jpg "Center"
[l]: ./doc_imgs/l.jpg "Left"
[r]: ./doc_imgs/r.jpg "Right"
[cp]: ./doc_imgs/c-processed.jpg "Center"
[lp]: ./doc_imgs/l-processed.jpg "Left"
[rp]: ./doc_imgs/r-processed.jpg "Right"

[shadow]: ./doc_imgs/shadow.JPG "Shadow training"


## Files Submitted

My project includes the following files:

- model.py containing the script to create and train the model
- drive.py for driving the car in autonomous mode
- model.h5 containing a trained convolution neural network
- README.md summarizing the results
- run1.mp4 showing how the car drove in autonomous mode

To run the code:

1. Download the car simulator (provided by Udacity):
  - [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip)
  - [MacOS](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip)
  - [Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip)
2. Run the simulator and drive.py with the code `python drive.py model.h5`
3. Choose "Autonomous Mode" from the simulator

## Model Architecture and Training Strategy

### Model Architecture

I used a similar CNN structure with that of [NVIDIA End-to-End Learning](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

![NVIDIA Net][nvidianet]

I made two adjustments to the architecture in this project:

1. The size of inputs is adjusted to use the recorded 65 x 320 pixels image. Depths, kernels, and strides are however exactly the same.
2. Dropout layers were added before and after each convolutional layer. More on this in the next section.

Here is the architecture in more detail:

| Layer                     |     Description                               | 
|:-------------------------:|:---------------------------------------------:| 
| Input                     | 65x320x3 RGB image | 
| Dropout                   | Drop 0.1  |
| Convolution 5x5           | 2x2 stride, valid padding, outputs 31x158x24 |
| RELU                      |                       |
| Dropout                   | Drop 0.25 |
| Convolution 5x5           | 2x2 stride, valid padding, outputs 14x77x36 |
| RELU                      |                       |
| Dropout                   | Drop 0.25 |
| Convolution 5x5           | 2x2 stride, valid padding, outputs 5x37x48 |
| RELU                      |                       |
| Dropout                   | Drop 0.50 |
| Convolution 3x3           | 1x1 stride, valid padding, outputs 3x35x64 |
| RELU                      |                       |
| Dropout                   | Drop 0.50 |
| Convolution 3x3           | 1x1 stride, valid padding, outputs 1x33x64 |
| RELU                      |                       |
| Dropout                   | Drop 0.50 |
| Fully connected (flatten) | outputs 2112 |
| Fully connected           | outputs 100 |
| Fully connected           | outputs 10 |
| Fully connected           | outputs 1 |


For the preprocessing step, here are the steps that I did:

1. I cropped 70px from the top of the image to remove the parts above the horizon and 25px from the bottom to remove the hood of the car. **Camera Positioning** section below contains images from the cameras that illustrate how the car's hood was shown.
3. And finally, I normalized the images.

The paper suggests using YUV color space. YUV color space is useful for detecting real-world images as described in its Wikipedia's definition in [here](https://en.wikipedia.org/wiki/YUV):

> YUV is a color space typically used as part of a color image pipeline. It encodes a color image or video taking human perception into account, ...

And [here](https://en.wikipedia.org/wiki/YUV#Luminance.2Fchrominance_systems_in_general):

> ... Understanding this human shortcoming, standards such as NTSC and PAL reduce the bandwidth of the chrominance channels considerably.

These benefits do not apply in this project since we are using image data from the simulation, so I decided to keep using RGB color space.

### Attempts to Reduce Overfitting in The Model

I attempted to reduce overfitting by using Dropout layers with settings as described in [this paper](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf) page 1938: Street View House Numbers. That is, dropout rate of 0.1, 0.25, 0.25, 0.5, 0.5, 0.5 going from input to convolutional layers.

To test if Dropbox actually works, I trained with and without Dropout layers and compared the results. See Appendix I for more details.

Based on the experiment results, I decided to incorporate Dropout into the model architecture. It was really interesting to see how effective Dropout was in reducing model overfitting in this project.

### Model Parameter Tuning

[Adam Optimizer](https://keras.io/optimizers/#adam) was used with the default hyperparameter settings:

- Learning Rate: 0.001
- Beta 1: 0.9
- Beta 2: 0.999
- Epsilon: 10^(-8)
- Decay: 0.0

Other training parameters:

- Epoch: 5
- Batch Size: 32

### Training Data Generation

#### Camera Positioning

In the car, three cameras were placed on top, left, and right of the car, all facing ahead, giving us three separate views: center, left, and right.

The idea was to train the car to drive by feeding it images and their respective steering rotation (positive values being clockwise or in other words turning right). The left and right positioned cameras gave some additional insights to the car what it would look like if the car was slightly tilting to the left and right (explained further in the **Data Augmentation** section below.

Here is an example of left, center, and right images:

![Left][l]
![Center][c]
![Right][r]

And after the processing step:

![Left Processed][lp]
![Center Processed][cp]
![Right Processed][rp]


#### Training Sessions

The very first training I did was only to see whether the model works. I did this by getting three sets of observations: one with steering angle of 0.0 (moving forward), another with steering angle of > 0.0 (turning right) and one more with steering angle of < 0.0 (turning left). This training session is named **Overfit** session and it was not included in the final model.

The baseline training session was the **Iterative** session where the model was trained by first manually driving the car in center lane for two laps. The model is then saved, and further adjustments were added iteratively i.e. load the model and update the weights with additional training data. In total, there were 4,908 observations (or 14,724 images if we count center, left, and right images on their own).

After the iterative session completed, the car drove without going off-lane in about 3 laps, but it did so in a zig-zag route, until eventually went for the hills.

The car was then trained to tackle extreme cases such as when the car is facing towards the edge of the road by doing a **Recovery Lap** training session. There were 470 observations or 1,410 images in this session.

The car was able to consistently stay on track, but I noticed the car was avoiding the shadows on the street. This was undesirable, so I added another training data by driving the car past a part of the road covered by shadows (197 tobservations, 591 images):

![shadow][shadow]

The car was able to brave through shadows, but unfortunately, it zig-zagged its way to the lake. The most likely explanation here is the model overfitted the shadow training data that it forgot how to handle edge cases.

One way this could have been corrected is by re-training the shadow training session with lower alpha, but I decided to do another large set of training. The reason for doing so was to see if it is possible not to have to curate the training data or adjust the hyperparameters too much, and instead just drive more often and the car would learn on its own.

For the last training data (**Reverse** session), I drove the car in the reverse direction for nearly two laps and updated the model. There were 4,008 observations in total in this step (12024 images).

And finally, the car drove relatively smoothly through track 1 as shown in run1.mp4 video. It did not perform so well on track 2 yet, but this is something I will get back to after I completed this Nanodegree program, who knows if there is a technique taught in class specifically for this.

#### Data Augmentation

After gathering the observations, we augmented the training data by including the following data:

1. Left and right camera inputs were paired with adjusted current wheel measurement. For the left camera input I added `+0.2` and `-0.2` for the right camera input. It can be translated as giving the car the following instructions: "when you see the car is tilting to the left, steer the wheel to the right by 0.2, and vice versa". This means we get two more data points from a single observation.
2. Duplicate all observations by flipping both their image and steering measurement. This technique ensures we have equal number of left and right turning training examples.

#### Hardware

Initially, I used a standard keyboard to drive the car. It did not go well since it was not possible to keep the wheel rotation at a certain radius. It resulted in a driving agent that zig-zagged the car to its demise (either crashing into a hill or falling into the lake).

One of the largest improvements was the result of simply changing the input hardware to use a Joystick (I used Extreme 3D Pro from Logitech). The better input data made it possible to create a model (more easily) that keeps the car on the road.

---

## Appendix I. Dropout Experiment

I experimented by training each training session both with and without dropout layers incorporated into the model architecture. As explained in section **Training Sessions** above, the sessions are divided into two parts: "overvit" and "iterative" training sessions. From all the results showcased below, it was perfectly clear that Dropout layers does help reducing overfitting i.e. the gaps between training and validation loss got smaller in all cases.

### No dropout - overfit

![No Dropout - Overfit Session][no_dropout-overfit]

```
Epoch 1/5
2/2 [==============================] - 2s - loss: 0.3206 - val_loss: 0.6727
Epoch 2/5
2/2 [==============================] - 0s - loss: 0.0949 - val_loss: 0.8449
Epoch 3/5
2/2 [==============================] - 0s - loss: 0.3083 - val_loss: 0.7571
Epoch 4/5
2/2 [==============================] - 0s - loss: 0.0734 - val_loss: 0.6789
Epoch 5/5
2/2 [==============================] - 0s - loss: 0.1627 - val_loss: 0.6613
```

Notice the large gap between testing and validation loss.

### No dropout - iterative

![No Dropout - Iterative Session][no_dropout-iterative]

```
Epoch 1/5
80/80 [==============================] - 25s - loss: 0.0108 - val_loss: 0.0331
Epoch 2/5
80/80 [==============================] - 22s - loss: 4.3192e-05 - val_loss: 0.0342
Epoch 3/5
80/80 [==============================] - 22s - loss: 9.2005e-07 - val_loss: 0.0339
Epoch 4/5
80/80 [==============================] - 23s - loss: 1.9444e-04 - val_loss: 0.0346
Epoch 5/5
80/80 [==============================] - 23s - loss: 9.9103e-05 - val_loss: 0.0347
```

The validation loss did not even decrease when the model went through more training epochs. Gap sizes were more or less similar with the Overfit training session.

### Dropout - overfit

![Dropout - Overfit Session][dropout-overfit]

```
Epoch 1/5
2/2 [==============================] - 2s - loss: 0.3252 - val_loss: 0.6169
Epoch 2/5
2/2 [==============================] - 0s - loss: 0.2808 - val_loss: 0.4617
Epoch 3/5
2/2 [==============================] - 0s - loss: 0.2221 - val_loss: 0.2430
Epoch 4/5
2/2 [==============================] - 0s - loss: 0.1803 - val_loss: 0.4088
Epoch 5/5
2/2 [==============================] - 0s - loss: 0.2159 - val_loss: 0.3956
```

Gaps between training and validation were smaller than the Non-Dropout case, but the bounce-back of validation errors indicates that the model may require more training data.


### Dropout - iterative

![Dropout - Iterative Session][dropout-iterative]

```
Epoch 1/5
80/80 [==============================] - 26s - loss: 0.0415 - val_loss: 0.0184
Epoch 2/5
80/80 [==============================] - 23s - loss: 0.0242 - val_loss: 0.0180
Epoch 3/5
80/80 [==============================] - 23s - loss: 0.0201 - val_loss: 0.0167
Epoch 4/5
80/80 [==============================] - 24s - loss: 0.0155 - val_loss: 0.0126
Epoch 5/5
80/80 [==============================] - 24s - loss: 0.0126 - val_loss: 0.0127
```

This is a curious case, since validation loss scores were lower than training losses. Perhaps there is an issue with how the validation dataset were selected, but otherwise we see the gaps between training and validation losses shorten over time, so we may conclude that Dropbox does indeed help in reducing overfitting, at least in this case.
