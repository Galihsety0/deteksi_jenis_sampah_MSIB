from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, GlobalAveragePooling2D
import keras.applications.mobilenet_v2 as mobilenetv2
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
import tensorflow.keras as keras
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda, GlobalAveragePooling2D, Dense
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input



# IMAGE_WIDTH = 320    
# IMAGE_HEIGHT = 320
# IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
# IMAGE_CHANNELS = 3

# categories = {0: 'paper', 1: 'cardboard', 2: 'plastic', 3: 'metal', 4: 'trash', 5: 'battery',
#               6: 'shoes', 7: 'clothes', 8: 'green-glass', 9: 'brown-glass', 10: 'white-glass',
#               11: 'biological'}


def make_model():
    # Membangun model
    model = tf.keras.Sequential()
    
    # Menambahkan lapisan Lambda
    model.add(tf.keras.layers.Lambda(preprocess_input, input_shape=(320, 320, 3)))

    # Menambahkan lapisan MobilenetV2
    model.add(tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet', pooling='avg'))
    
    # Menambahkan lapisan Flatten
    model.add(tf.keras.layers.Flatten())
    
    # Menambahkan lapisan Dense
    model.add(tf.keras.layers.Dense(12, activation='softmax'))
    
    return model


# def make_model():
#     base_model = mobilenetv2.MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT,IMAGE_CHANNELS))
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(1024, activation='relu')(x)
#     predictions = Dense(12, activation='softmax')(x)
#     model = Model(inputs=base_model.input, outputs=predictions)
    
#     return model

# def preprocess_image(image):
#     img = image.resize((224, 224))
#     img = img.convert('RGB')
#     img = img_to_array(img)
#     img = np.expand_dims(img, axis=0)
    
#     return mobilenetv2.preprocess_input(img)

# def make_model ():
#     mobilenetv2_layer = mobilenetv2.MobileNetV2(include_top = False, input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT,IMAGE_CHANNELS),
#                        weights = 'imagenet')
#     model = Sequential()
#     model.add(keras.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
#     model.add(Lambda(mobilenetv2_preprocessing))
#     model.add(mobilenetv2_layer)
#     model.add(tf.keras.layers.GlobalAveragePooling2D())
#     model.add(Dense(len(categories), activation='softmax'))
#     #create a custom layer to apply the preprocessing
#     def mobilenetv2_preprocessing(img):
    
#     return mobilenetv2.preprocess_input(img)