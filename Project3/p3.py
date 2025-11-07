import os
import numpy as np
import cv2
import gc
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from scipy import stats
from pathlib import Path, PureWindowsPath


def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))
    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows
    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list


def get_tiny_image(img, output_size):
    feature = None
    # To do    
    return feature


def predict_knn(feature_train, label_train, feature_test, k):
    label_test_pred = None
    # To do    
    return label_test_pred


def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    confusion, accuracy = None, None
    # To do    
    return confusion, accuracy


def compute_dsift(img, stride, size):
    dense_feature = None
    # To do    
    return dense_feature


def build_visual_dictionary(dense_feature_list, dict_size):
    vocab = None
    # To do

    # You might want to save the current vocab if you think it is good.
    np.save("vocab.npy", vocab)
    return vocab


def compute_bow(feature, vocab):
    bow_feature = None
    # To do
    return bow_feature


def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list, vocab=None):
    confusion, accuracy = None, None
    # To do
    return confusion, accuracy


def predict_svm(feature_train, label_train, feature_test):
    label_test_pred = None
    # To do    
    return label_test_pred


def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list, vocab=None):
    confusion, accuracy = None, None
    # To do
    return confusion, accuracy


def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':

    vocab = None
    label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info("./scene_classification_data")
    
    confusion, accuracy = classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    del confusion, accuracy
    gc.collect()

    # vocab = np.load("vocab.npy")
    confusion, accuracy = classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list, vocab)
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    del confusion, accuracy
    gc.collect()

    # vocab = np.load("vocab.npy")
    confusion, accuracy = classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list, vocab)
    visualize_confusion_matrix(confusion, accuracy, label_classes)
