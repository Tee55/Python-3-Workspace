import numpy as np
from PIL import Image
import glob
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

#Define Directories for train, test & Validation Set
train_path = 'dataset/train'
test_path = 'dataset/test'
valid_path = 'dataset/val'

batch_size = 10

'''
img_height = 299
img_width = 299
'''

img_height = 224
img_width = 224

def main():

    image_gen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
    test_data_gen = ImageDataGenerator(rescale = 1./255)

    train = image_gen.flow_from_directory(train_path, target_size=(img_height, img_width), color_mode='grayscale', class_mode='binary', batch_size=batch_size)
    test = test_data_gen.flow_from_directory(test_path, target_size=(img_height, img_width), color_mode='grayscale', shuffle=False, class_mode='binary', batch_size=batch_size)
    valid = test_data_gen.flow_from_directory(valid_path, target_size=(img_height, img_width), color_mode='grayscale', class_mode='binary', batch_size=batch_size)

    cnn = create_model()

    early = EarlyStopping(monitor='val_loss', mode='min', patience=3)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.3, min_lr=0.000001)

    callbacks_list = [early, learning_rate_reduction]

    weights = compute_class_weight('balanced', np.unique(train.classes), train.classes)
    cw = dict(zip( np.unique(train.classes), weights))

    cnn.fit(train, epochs=25, validation_data=valid, class_weight=cw, callbacks=callbacks_list)

    test_accu = cnn.evaluate(test)
    print('The testing accuracy is :', test_accu[1]*100, '%')

    preds = cnn.predict(test, verbose=1)

    predictions = preds.copy()
    predictions[predictions <= 0.5] = 0
    predictions[predictions > 0.5] = 1

    cm = pd.DataFrame(data=confusion_matrix(test.classes, predictions, labels=[0, 1]),index=["Actual Normal", "Actual Covid"],
    columns=["Predicted Normal", "Predicted Covid"])
    sns.heatmap(cm, annot=True, fmt="d")

    plt.show()

def decom():

    image_gen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
    test_data_gen = ImageDataGenerator(rescale = 1./255)

    train = image_gen.flow_from_directory(train_path, target_size=(img_height, img_width), color_mode='rgb', class_mode='binary', batch_size=batch_size)
    test = test_data_gen.flow_from_directory(test_path, target_size=(img_height, img_width), color_mode='rgb', shuffle=False, class_mode='binary', batch_size=batch_size)
    valid = test_data_gen.flow_from_directory(valid_path, target_size=(img_height, img_width), color_mode='rgb', class_mode='binary', batch_size=batch_size)

    decom = create_decomp()

    features = decom.predict(train)

    pca = PCA()

    # prepare transform on dataset
    pca.fit(features)

    # apply transform to dataset
    transformed = pca.transform(features)

    print(transformed.shape)

    pred_y = k_meanClus(transformed)

    class_comp(pred_y)

def class_comp(pred_y):

    image_gen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
    test_data_gen = ImageDataGenerator(rescale = 1./255)

    train = image_gen.flow_from_directory(train_path, target_size=(img_height, img_width), color_mode='rgb', class_mode='binary', batch_size=batch_size)
    test = test_data_gen.flow_from_directory(test_path, target_size=(img_height, img_width), color_mode='rgb', shuffle=False, class_mode='binary', batch_size=batch_size)
    valid = test_data_gen.flow_from_directory(valid_path, target_size=(img_height, img_width), color_mode='rgb', class_mode='binary', batch_size=batch_size)

    cnn = create_model()

    early = EarlyStopping(monitor='val_loss', mode='min', patience=3)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.3, min_lr=0.000001)

    callbacks_list = [early, learning_rate_reduction]

    weights = compute_class_weight('balanced', np.unique(train.classes), train.classes)
    cw = dict(zip( np.unique(train.classes), weights))

    cnn.fit(pred_y, epochs=25, validation_data=valid, class_weight=cw, callbacks=callbacks_list)

    test_accu = cnn.evaluate(test)
    print('The testing accuracy is :', test_accu[1]*100, '%')

    preds = cnn.predict(test, verbose=1)

    predictions = preds.copy()
    predictions[predictions <= 0.5] = 0
    predictions[predictions > 0.5] = 1

    cm = pd.DataFrame(data=confusion_matrix(test.classes, predictions, labels=[0, 1]),index=["Actual Normal", "Actual Covid"],
    columns=["Predicted Normal", "Predicted Covid"])
    sns.heatmap(cm, annot=True, fmt="d")

    plt.show()

def create_model():

    cnn = Sequential()

    cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Conv2D(64, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Conv2D(64, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
    cnn.add(MaxPooling2D(pool_size = (2, 2)))
    cnn.add(Flatten())
    cnn.add(Dense(activation = 'relu', units = 128))
    cnn.add(Dense(activation = 'relu', units = 64))
    cnn.add(Dense(activation = 'sigmoid', units = 1))
    cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return cnn

def create_decomp():

    model = VGG19(weights='imagenet')

    return model

def k_meanClus(X):
    '''
    wcss = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    '''

    kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred_y = kmeans.fit_predict(X)

    '''
    plt.scatter(X[:,0], X[:,1])
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
    plt.show()
    '''

    print(pred_y)

    return pred_y

if __name__ == '__main__':
    #main()
    decom()
    