import numpy as np
import config
from custom_resnet import get_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import pickle

img_path = config.PREDICT_DIR + "1.jpg"
svm_path = None


def test_predict():
    model = get_model()

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)

    clf_svm = pickle.load(open(svm_path, 'rb'))

    y_pred = clf_svm.predict(features)

    print(y_pred)

if __name__ == '__main__':
    test_predict()