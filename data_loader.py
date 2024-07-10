import numpy as np
from keras.utils import np_utils

def load_data(numpy_save_folder, fit_type='float32'):
    train_feature = np.load(numpy_save_folder + 'train_feature.npy')
    test_feature = np.load(numpy_save_folder + 'test_feature.npy')
    train_label = np.load(numpy_save_folder + 'train_label.npy')
    test_label = np.load(numpy_save_folder + 'test_label.npy')
    class_weights = np.load(numpy_save_folder + 'class_weights.npy')

    train_feature_vector = train_feature.reshape(len(train_feature), 100, 120, 1).astype(fit_type)
    test_feature_vector = test_feature.reshape(len(test_feature), 100, 120, 1).astype(fit_type)

    train_feature_normalize = train_feature_vector / 255
    test_feature_normalize = test_feature_vector / 255

    train_label_onehot = np_utils.to_categorical(train_label, 26)
    test_label_onehot = np_utils.to_categorical(test_label, 26)

    return (train_feature_normalize, test_feature_normalize,
            train_label_onehot, test_label_onehot, class_weights)
