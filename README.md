# WAME Optimizer
Implementation of the WAME optimization algorithm as described in the paper [Training Convolutional Networks with Weight-wise Adaptive Learning Rates](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2017-50.pdf) by Mosca and Magoulas. Implemented as an [optimizer class](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers) for TensorFlow 2.0 or higher.

#### Paper Abstract
Current state–of–the–art Deep Learning classification with Convolutional Neural Networks achieves very impressive results, which are, in some cases, close to human level performance. However, training these methods to their optimal performance requires very long training periods, usually by applying the Stochastic Gradient Descent method. We show
that by applying more modern methods, which involve adapting a different learning rate for each weight rather than using a single, global, learning rate for the entire network, we are able to reach close to state–of–the–art performance on the same architectures, and improve the training time and accuracy.

#### Usage
The optimizer class is only compatible with tensorflow>=2.0.  
Call WAME when compiling a TensorFlow model, see below example using the Keras API:

``` python
from tensorflow import keras
from wame import WAME

model = keras.models.Sequential([
  keras.layers.Flatten(input_shape=(28, 28)),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dropout(0.2),
  keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=WAME(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

```
