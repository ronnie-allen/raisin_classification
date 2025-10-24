# Raisin Classification with CNN

This project trains a Convolutional Neural Network (CNN) to classify raisins into three categories: RAISIN_BLACK, RAISIN_GRADE1, and RAISIN_PREMIUM using TensorFlow/Keras.

## Dataset

The dataset consists of approximately 960 images per class (total: 2880 images) organized in folders by class.

## Requirements

Install the dependencies using:

```
pip install -r requirements.txt
```

## Files

- `raisin_classification.ipynb`: Jupyter notebook with data loading, model training, evaluation, and prediction.
- `raisin_cnn_model.h5`: Trained model file (after running the notebook).
- `requirements.txt`: Python dependencies.
- `README.md`: This file.

## Usage

### 1. Data Preparation

The notebook automatically handles data preparation using TensorFlow's ImageDataGenerator for loading and augmenting the images.

### 2. Training

Run the notebook cells sequentially to train the model:

- Load and preprocess data.
- Build the CNN model.
- Train with data augmentation and callbacks (early stopping, learning rate reduction).
- Evaluate on validation set with confusion matrix.

### 3. Prediction

Use the trained model to classify new images:

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load model
model = load_model('raisin_cnn_model.h5')

# Load image
img = load_img('path/to/image.jpg', target_size=(150, 150))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
class_names = ['RAISIN_BLACK', 'RAISIN_GRADE1', 'RAISIN_PREMIUM']
predicted_class = class_names[np.argmax(predictions)]

print(f'Predicted: {predicted_class}')
```

## Model Architecture

- Convolutional layers with max pooling.
- Dropout for regularization.
- Dense layers for classification.
- Softmax output for 3 classes.

## Training Boosting Techniques

- Data augmentation (rotations, zooms, flips).
- Early stopping to prevent overfitting.
- Learning rate reduction on plateau.

## Workflow Diagram

```mermaid
graph TD
    A[Dataset: RAISIN-DATASET with 3 classes] --> B[Preprocessing: Resize to 150x150, Augment (rotate, zoom, flip)]
    B --> C[Build CNN Model: Conv2D layers, MaxPooling, Dropout, Dense]
    C --> D[Training: Adam optimizer, Early stopping, LR reduction]
    D --> E[Evaluation: Validation accuracy, Confusion Matrix]
    E --> F[Export Model: Save as H5 file]
    F --> G[Prediction: Load model, Preprocess image, Classify]
```

## Performance

Expected validation accuracy: ~70-85% depending on training.

## Notes

- Ensure the dataset directory `RAISIN-DATASET/` is in the project root.
- The model is saved as `raisin_cnn_model.h5` after training.
