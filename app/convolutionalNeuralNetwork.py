import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np

IMG_LEN = 224
IMG_SHAPE = (IMG_LEN, IMG_LEN, 3)
N_BREEDS = 120


def loadDataset():
    dataset, info = tfds.load(name="stanford_dogs", with_info=True)
    get_name = info.features['label'].int2str
    training_data = dataset['train']
    test_data = dataset['test']
    return training_data, test_data


def preprocess(ds_row):
    # Image conversion int->float + resizing
    image = tf.image.convert_image_dtype(ds_row['image'], dtype=tf.float32)
    image = tf.image.resize(image, (IMG_LEN, IMG_LEN), method='nearest')

    # Onehot encoding labels
    label = tf.one_hot(ds_row['label'], N_BREEDS)

    return image, label


def prepare(dataset, batch_size=None):
    ds = dataset.map(preprocess, num_parallel_calls=4)
    ds = ds.shuffle(buffer_size=1000)
    if batch_size:
        ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds



def createModel():
    training_data, test_data = loadDataset()
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(N_BREEDS, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adamax(0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'top_k_categorical_accuracy'])

    model.summary()

    train_batches = prepare(training_data, batch_size=32)
    test_batches = prepare(test_data, batch_size=32)
    return model, train_batches, test_batches


def fitModelAndSave():
    model, train_batches, test_batches = createModel()
    history = model.fit(train_batches,
                        epochs=30,
                        validation_data=test_batches)


    model.save('modelDogsBreeds2.h5')

    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Test accuracy')
    plt.legend()

    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Test accuracy')
    plt.legend()
    plt.savefig('lc.svg')


from PIL import Image, ImageOps


def classifyImage(path):
    with open('static/txt/breeds.txt', 'r') as reader:
        breeds = []
        for breed in reader:
            try:
                breed = breed.replace('_', ' ')
                breed = breed.title()
            except:
                breed = breed.title()
            breeds.append(breed[:-1])

    model = keras.models.load_model('modelDogsBreeds.h5')

    data = np.ndarray(shape=(1, IMG_LEN, IMG_LEN, 3), dtype=np.float32)
    img = Image.open(path) #np. 'static/jpgs/file.jpg'
    image = img
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # Turning the image into a numpy array, normalizing and loading the image
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 255)
    data[0] = normalized_image_array

    # Prediction
    prediction = model.predict(data)

    return breeds[np.argmax(prediction)]
