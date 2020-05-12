#based on this work
#https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model

class GradCAM():
  def __init__(self, inputs, last_layer, output):

      self.inputs = inputs
      self.last_layer = last_layer
      self.output = output

      self.model = Model(inputs=[self.inputs],
                  outputs=[self.last_layer ,self.output])

  def globalAveragePooling(self, classIdx, inputs):

      with tf.GradientTape() as tape:
        input_cast = tf.cast(inputs, tf.float32)
        (convOutputs, pred) = self.model(input_cast)
        loss = pred[:, classIdx]

      grads = tape.gradient(loss, convOutputs)#the gradient of the score for class classIdx with respect to feature maps convOutputs
      convOutputs = convOutputs[0]
      grads = grads[0]

      alpha = tf.reduce_mean(grads, axis=(0, 1))

      return alpha, convOutputs, grads

  def ReLU(self, inp):
      return np.where(inp > 0, inp, 0)

  def computeLGRAD_CAM(self, classIdx, inputs):

      alpha, convOutputs, grads = self.globalAveragePooling(classIdx, inputs)
      Lcam = tf.reduce_sum(tf.multiply(alpha, convOutputs), axis=-1)
      Lcam = self.ReLU(Lcam)

      return Lcam, convOutputs, grads
