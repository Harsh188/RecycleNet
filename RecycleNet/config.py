PACKAGE_ROOT = "./"
DATASET_DIR = PACKAGE_ROOT + "dataset"
TRAIN_DIR = DATASET_DIR + "/train"
TEST_DIR = DATASET_DIR + "/test"

TRAINED_MODEL_DIR_RESNET = PACKAGE_ROOT + "trained_models/partial"
TRAINED_MODEL_DIR_SVM = PACKAGE_ROOT + "trained_models/ResNet_SVM/"
LOGS_DIR = PACKAGE_ROOT + "logs"

GRAPHDIR_CM = "./imgs/graphs/confusion_matrix/"

PREDICT_DIR = "./imgs"

"""
Hyper-parameters for Resnet50
"""
EPOCHS = 12
STEPS_PER_EPOCH = 30
BATCH_SIZE = 32

"""
Hyper-parameters for SVM
"""
C = 1000
GAMMA = 0.5
sample_count = 1050

IMG_WIDTH = 224
IMG_HEIGHT = 224