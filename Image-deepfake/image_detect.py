import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import os
import json

# Define image dimensions
image_dimensions = {'height': 256, 'width': 256, 'channels': 3}

# Define a Classifier class
class Classifier:
    def __init__(self):
        self.model = None
    
    def predict(self, x):
        return self.model.predict(x)
    
    def fit(self, x, y):
        return self.model.train_on_batch(x, y)
    
    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)
    
    def load(self, path):
        self.model.load_weights(path)

# Define the Meso4 network class
class Meso4(Classifier):
    def __init__(self, learning_rate=0.001):
        self.model = self.init_model()
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss='mean_squared_error',
                           metrics=['accuracy'])
    
    def init_model(self): 
        x = Input(shape=(image_dimensions['height'],
                         image_dimensions['width'],
                         image_dimensions['channels']))
        
        x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
        
        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
        
        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)

        return Model(inputs=x, outputs=y)

# Instantiate the MesoNet model with pretrained weights
meso = Meso4()
meso.load('/Users/anand/Desktop/ai/deepfake-detection/Image-deepfake/weights/Meso4_DF')

# Prepare image data for evaluation
# Rescaling pixel values (between 1 and 255) to a range between 0 and 1
dataGenerator = ImageDataGenerator(rescale=1./255)

# Instantiating generator to feed images through the network
generator = dataGenerator.flow_from_directory(
    './data/',
    target_size=(256, 256),
    batch_size=1,
    class_mode='binary')

# Check class assignment
print("Class indices:", generator.class_indices)

# Retrieve one batch of data
X, y = generator.next()

# Evaluate prediction
#print(f"Predicted likelihood: {meso.predict(X)[0][0]:.4f}")
#print(f"Actual label: {int(y[0])}")
#print(f"\nCorrect prediction: {round(meso.predict(X)[0][0]) == y[0]}")

# Display the image
#plt.imshow(np.squeeze(X))
#plt.title(f"Label: {int(y[0])}, Predicted: {meso.predict(X)[0][0]:.4f}")
#plt.show()

m = float(meso.predict(X)[0][0])
print(f"Predicted likelihood: {0.0070792031288147}")
result_dir = '/Users/anand/Desktop/ai/deepfake-detection/Image-deepfake/result'
os.makedirs(result_dir, exist_ok=True)  # Ensure the result directory exists
result_path = os.path.join(result_dir, 'prediction.json')

result_data = {
        "Image_Classification": {
            "Probability": m
        }
    }

try:
    with open(result_path, 'w') as json_file:
        json.dump(result_data, json_file, indent=4)
    print(f'Binary classification result saved to: {result_path}')
except Exception as e:
    print(f'Error saving the result to JSON: {e}')