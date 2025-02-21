# Application of NAS to Classification Problem

(*The data used in the experiment are specially selected for computational convenience and understanding of the experiment.*)

## Overview

This project focuses on searching for an optimal neural network architecture for classifying fashion products into 10 categories. The dataset used for training and evaluation is **Fashion-MNIST**, which consists of **70,000 grayscale images** (28x28 pixels) with corresponding labels from 10 categories.

The training pipeline is optimized using **Neural Architecture Search (NAS)**.

---

## Dataset

- **Fashion-MNIST** dataset is used, containing:
  - 70,000 images (28x28 grayscale)
  - 10 different fashion categories

---

## Model Training Pipeline

### 1. Load Data
```python
from tensorflow import keras
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.datasets import fashion_mnist
from keras.utils import to_categorical

NUM_CLASSES = 10

def load_data():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)
    
    return (x_train, x_test), (y_train, y_test)
```

### 2. Build Model
```python
def build_model():
    model = Sequential([
        Conv2D(32, 3, activation='relu'),
        Conv2D(64, 3, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
```

### 3. Train Model
```python
(x_train, x_test), (y_train, y_test) = load_data()
model = build_model()

model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_test, y_test))

loss, acc = model.evaluate(x_test, y_test)
print(f"Accuracy: {acc}")
```

**Result:** The model achieves an accuracy of **~89.8%**.

---

## Neural Architecture Search (NAS)

### Defining the Search Space
To find a better architecture, a search space is defined containing multiple possible configurations:

```python
search_space = {
    "filter_size_c1": {"_type": "choice", "_value": [32, 64, 128]},
    "filter_size_c2": {"_type": "choice", "_value": [32, 64, 128]},
    "kernel_size_c1": {"_type": "choice", "_value": [3, 5]},
    "kernel_size_c2": {"_type": "choice", "_value": [3, 5]},
    "nb_units": {"_type": "choice", "_value": [80, 100, 120]},
    "learning_rate": {"_type": "uniform", "_value": [0.001, 0.01]}
}
```

### Running NAS Experiment

```python
from nni.experiment import Experiment

experiment = Experiment('local')
experiment.config.trial_command = 'python fashionmnistmodel.py'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.trial_concurrency = 1
experiment.config.max_trial_number = 5

experiment.run()
```

### Results
After running NAS, the best architecture achieves an accuracy of **~91.4%**, which is an improvement over the manually designed model.

---

## Bibliography
- [YouTube Lecture](https://www.youtube.com/watch?v=td820ts6gUU&t=3687s)

---

## Conclusion
Using **Neural Architecture Search (NAS)** with **Neural Network Intelligence (NNI)**, we achieved a higher accuracy in classification compared to a manually designed architecture. The automated search process efficiently explored different model configurations to find the best-performing one.

