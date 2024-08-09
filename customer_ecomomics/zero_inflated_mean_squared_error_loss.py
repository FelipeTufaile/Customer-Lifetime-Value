# The MIT License (MIT)
#
# Copyright (c) 2024 Felipe Solla Tufaile
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ============================================================================

## Loading dependencies
##############################################################################################################
# Importing Tensorflow and Tensorflow layers 
import tensorflow as tf
import tensorflow.keras.layers as tfl

## Defining cost function: Zero-Inflated Mean Squared Error (ZIMSE)
##############################################################################################################
def zero_inflated_mean_squared_error_loss(y_true:tf.Tensor, y_pred:tf.Tensor) -> tf.Tensor:
  """Computes the zero-inflated mean squared error loss.

  Note: In order to really leverage the capabilities of the Zero-Inflated Mean Squared Error model make sure
  that the customer lifetime value labels are as close as possible to a normal distribution. In case it is not
  normally distributed, it is advisable to transaform the feature using the function y = np.log(x + 1).
  In case the previous transformation is implemented, you may want to apply the following inverse transformation
  to restore customer lifetime values: x = np.exp(y). It is not necessary to remove the unit ("-1") in the inverse
  function. The model itself is capable of abstracting this constant during training.

  Usage with tf.keras API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.compile('Adam', loss=zero_inflated_mean_squared_error_loss)
  ```

  Arguments:
    y_true [tf.Tensor]: True targets, tensor of shape [batch_size, 1].
    y_pred [tf.Tensor]: Tensor of output layer, tensor of shape [batch_size, 2].

  Returns:
    Zero-inflated mean squared error loss value.
  """

  ## Creating a tensor from y_true.
  # This will a tensor with shape (m, 1), where "m" is the number of samples
  tf_true = tf.convert_to_tensor(y_true, dtype=tf.float32)

  ## Creating a tensor from y_pred.
  # This will a tensor with shape (m, n), where "m" is the number of samples and "n" is the number of outputs.
  # Output 1: probability of transaction (calculated from sigmoid activation function)
  # Output 2: Customer Lifetime Value (calculated from linear activation function)
  tf_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

  ## Calculating true labels.
  # This is necessary for the classification part of the cost function:
  # We want to distinguish CLV greater than 0 from CLV equal to zero.
  true_labels = tf.cast(tf_true > 0, tf.float32)

  ## Calculating predicted labels.
  # Transaction probabilities correspond to the first output of the y_pred tensor.
  pred_labels = tf_pred[..., :1]

  ## Classification loss
  # In order to calculate the classification loss we use Binary Cross-Entropy
  classification_loss = tf.keras.losses.binary_crossentropy(true_labels, pred_labels, from_logits=False)
  classification_loss = tf.reshape(classification_loss, [-1]) # Reshapes into a vector (m,)

  ## Adapting Customer Lifetime Value vector
  #pred_clv = tf.math.multiply(true_labels, tf_pred[..., 1:2])
  pred_clv = tf_pred[..., 1:2]

  ## Regression loss
  # In order to calculate the regression loss we use Mean-Squared Error
  regression_loss = tf.math.square(tf_true - pred_clv)
  regression_loss = tf.reshape(regression_loss, [-1]) # Reshapes into a vector (m,)

  ## Total Loss
  # Finally the total loss will be the sum of the classification loss and the regression loss
  total_loss = classification_loss + regression_loss

  return total_loss
