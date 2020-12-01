import numpy as np
import config
from custom_resnet import get_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import pickle

img_path = config.PREDICT_DIR + "cardboard.jpg"
svm_path = 'resnet_svm_2020-12-01-22:40.sav'


def test_predict():
    model = get_model()

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    # features = features.reshape(1,7*7*2048)

    clf_svm = pickle.load(open(svm_path, 'rb'))

    y_pred = clf_svm.score(features,[0])

    print(y_pred)

if __name__ == '__main__':
    test_predict()