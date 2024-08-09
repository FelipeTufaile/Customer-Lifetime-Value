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

# Importing custom layers
import zero_inflated_mean_squared_error_loss

## Defining model: Zero-Inflated Mean Squared Error Neural Network Model
##############################################################################################################
def zimse_model(
    input_size:int=None, 
    num_units:int=64, 
    learning_rate:float=1e-4, 
    emd_layers_dict:dict=None, 
    print_summary:bool=False
  ) -> keras.src.models.functional.Functional:
  """
    Function that creates a ZIMSE neural network model with fully connected layers and embedding layers.

    Parameters:
    ----------
    input_size: int
      The number of numerical features.
    num_units: int
      The number of units to be used in each fully connected layers.
    learning_rate: float
      The learning rate to be used in the optimizer.
    emd_layers_dict: dict
      An embedding dictionary containing:
      - Embedding layers names - Dictionary Keys;
      - Embedding layers dimensions (input, output) - Dictionary Values.
      Example:
        emd_layers_dict = {
          "month_layer": (month_input_dim, month_output_dim),
          "zipcode_layer": (zipcode_input_dim, zipcode_ouput_dim)
        }
    
    Returns:
    ----------
    model: keras.src.models.functional.Functional
      The ZIMSE Neural Network Model.
  """
  ## Initializing objects
  ##############################################################################
  # Defining an empty list of inputs
  inputs = []

  # Defining an empty list of layers
  layers = []
  
  ## Building Embedding Layers
  ##############################################################################
  # First check if the emd_layers_dict is empty
  if emd_layers_dict is None:
    pass
  else:
    # Iterating through all embedding layers in the dictionary 
    for emb_layer in emd_layers_dict:

      # Defining the input of the embedding layer
      emb_input = tfl.Input(shape=(1,), name=emb_layer)

      # Defining the embedding layer 
      emb_layer = tfl.Embedding(
          input_dim=emd_layers_dict[emb_layer][0],  # Defining input dimension
          output_dim=emd_layers_dict[emb_layer][1], # Defining output dimension
          trainable=True)(emb_input)

    # Reshaping the output of the embedding layer
    emb_output = tfl.Flatten()(emb_layer)

    # Adding the input to the inputs list
    inputs.append(emb_input)

    # Adding the ouput to the layers list
    layers.append(emb_output)

  ## Building Fully Connected Layers
  ##############################################################################

  # Defining input functional for numerial features
  num_input = tf.keras.Input(shape=(num_input_dim,), name='numerical_input')

  # Adding the numerical input to the inputs list
  inputs.append(num_input)

  # Adding the numerical input to the layers list
  layers.append(num_input)

  # Defining the initial fully connected layer
  L0 = tfl.Concatenate()(layers)

  # Defining hidden layer 1
  H1 = tfl.Dense(units=num_units, activation="relu")(L0)
  B1 = tfl.BatchNormalization()(H1)

  # Defining hidden layer 2
  H2 = tfl.Dense(units=num_units, activation="relu")(B1)
  B2 = tfl.BatchNormalization()(H2)

  # Defining hidden layer 3
  H3 = tfl.Dense(units=num_units, activation="relu")(H2)
  B3 = tfl.BatchNormalization()(H3)

  # Defining the propensity score unity (output layer)
  P4 = tfl.Dense(units=1, activation="sigmoid")(B3)

  # Defining the customer lifetime value unity (output layer)
  C4 = tfl.Dense(units=1, activation="relu")(B3)

  # Defining output layer
  L4 = tfl.concatenate([P4, C4])

  ## Building Final Neural Network Model
  ##############################################################################

  # Creating the neural network model
  model = tf.keras.Model(inputs=inputs, outputs=L4)

  # check if the model summary should be printed
  if print_summary:
    # summarize layers
    print(model.summary())

  # Defining optimizer
  optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

  # Compiling the model
  model.compile(loss=zero_inflated_mean_squared_error_loss, optimizer=optimizer)

  return model
