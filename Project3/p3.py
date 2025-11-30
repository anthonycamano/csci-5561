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
    '''
    input:
    img is a gray scale image, output_size=(w, h) is the size of the tiny image.

    output:
    feature is the tiny image representation by vectorizing the pixel intensity. The resulting size will be w×h.

    note: this part of the pdf did not mention using opencv to resize so i did it this way with numpy instead
    '''
    ## Step 1: get dimensions
    target_width, target_height = output_size
    orig_height, orig_width = img.shape
    
    ## Step 2: calc sampling indices
    row_indices = np.linspace(0, orig_height - 1, target_height).astype(int)
    col_indices = np.linspace(0, orig_width - 1, target_width).astype(int)
    
    ## Step 3: sample pixels using meshgrid indexing
    row_mesh, col_mesh = np.meshgrid(row_indices, col_indices, indexing='ij')
    resized_img = img[row_mesh, col_mesh]
    
    ## Step 4: flatten to 1D vector
    feature = resized_img.flatten()
    
    ## Step 5: convert to float to do math
    feature = feature.astype(np.float64)
    
    ## Step 6: normalize to zero mean
    feature = feature - np.mean(feature)
    
    ## Step 7: normalize to unit length -- also check for zero norm and set to 0
    norm = np.sqrt(np.sum(feature ** 2))
    if norm > 0:
        feature = feature / norm

    return feature


def predict_knn(feature_train, label_train, feature_test, k):
    label_test_pred = None
    '''
    input:

    output:
    label_test_pred is a n_te vector that specifies the predicted label for the testing data.
    
    note: i just used nearestneighbors from sklearn to do this part
    ''' 
    ## create and fit NearestNeighbors object
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto')
    nbrs.fit(feature_train)
    
    ## find k nearest neighbors
    distances, indices = nbrs.kneighbors(feature_test)
    
    ## get all neighbor labels at once
    neighbor_labels = label_train[indices]  # Shape: (n_test, k)
    
    ## mode finds the most common value along axis=1 (for each test sample)
    label_test_pred = stats.mode(neighbor_labels, axis=1, keepdims=False)[0]
    
    return label_test_pred


def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    confusion, accuracy = None, None
    '''
    input:


    output:


    note:
    '''
    # Step 1: Set parameters
    output_size = (16, 16)
    k = 3
    
    # Step 2: Extract training features
    n_train = len(img_train_list)
    feature_train = []
    
    print("Extracting training features...")
    for i, img_path in enumerate(img_train_list):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        feature = get_tiny_image(img, output_size)
        feature_train.append(feature)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_train} training images")
    
    feature_train = np.array(feature_train)
    
    # Step 3: Convert training labels to numeric
    label_to_index = {label: idx for idx, label in enumerate(label_classes)}
    label_train = np.array([label_to_index[label] for label in label_train_list])
    
    # Step 4: Extract test features
    n_test = len(img_test_list)
    feature_test = []
    
    print("Extracting test features...")
    for i, img_path in enumerate(img_test_list):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        feature = get_tiny_image(img, output_size)
        feature_test.append(feature)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_test} test images")
    
    feature_test = np.array(feature_test)
    
    # Step 5: Convert test labels to numeric
    label_test = np.array([label_to_index[label] for label in label_test_list])
    
    # Step 6: Predict using kNN
    print("Predicting labels with kNN...")
    label_test_pred = predict_knn(feature_train, label_train, feature_test, k)
    
    # Step 7: Build confusion matrix
    n_classes = len(label_classes)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    
    for true_label, pred_label in zip(label_test, label_test_pred):
        confusion[true_label, pred_label] += 1
    
    # Step 8: Calculate accuracy
    correct = np.trace(confusion)
    total = np.sum(confusion)
    accuracy = correct / total
    
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    return confusion, accuracy


def compute_dsift(img, stride, size):
    '''
    input:


    output:


    note:
    ''' 
    ## create SIFT object
    sift = cv2.SIFT_create()
    
    ## generate dense grid of keypoint locations
    height, width = img.shape
    keypoints = [cv2.KeyPoint(float(x), float(y), float(size))
                 for y in range(stride // 2, height, stride)
                 for x in range(stride // 2, width, stride)]
    
    ## compute SIFT descriptors at all locations
    _, descriptors = sift.compute(img, keypoints)
    
    ## handle edge case
    if descriptors is None:
        dense_feature = np.array([]).reshape(0, 128)
    else:
        dense_feature = descriptors
    return dense_feature


def build_visual_dictionary(dense_feature_list, dict_size):
    '''
    input:

    output:

    note:
    
    '''
    print(f"Pooling SIFT features from {len(dense_feature_list)} images...")
    all_features = np.vstack(dense_feature_list)
    print(f"Total SIFT features: {all_features.shape[0]}")
    
    ## sample features to speed up clustering
    max_features = 50000  ## use fewer features
    if all_features.shape[0] > max_features:
        print(f"Sampling {max_features} features from {all_features.shape[0]}...")
        indices = np.random.choice(all_features.shape[0], max_features, replace=False)
        all_features = all_features[indices]
    
    ## use faster K-means parameters
    print(f"Running K-means with {dict_size} clusters...")
    kmeans = KMeans(n_clusters=dict_size,
                    n_init=3,        ## fewer initializations
                    max_iter=100,    ## fewer iterations
                    random_state=42,
                    verbose=1)
    
    kmeans.fit(all_features)
    
    vocab = kmeans.cluster_centers_
    print(f"Visual dictionary built with shape: {vocab.shape}")
    
    ## save vocabulary
    np.save("vocab.npy", vocab)
    return vocab


def compute_bow(feature, vocab):
    '''
    input:

    output:

    note:
    
    '''
    dict_size = vocab.shape[0]
    n_features = feature.shape[0]
    
    ## handle 0
    if n_features == 0:
        return np.zeros(dict_size)
    
    ## use nearestNeighbors to find closest visual words
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto')
    nbrs.fit(vocab)
    
    ## find nearest visual word for each feature
    distances, indices = nbrs.kneighbors(feature)
    nearest_words = indices.flatten()
    
    ## build histogram
    bow_feature = np.bincount(nearest_words, minlength=dict_size).astype(float)
    
    ## normalize
    norm = np.linalg.norm(bow_feature)
    if norm > 0:
        bow_feature = bow_feature / norm

    return bow_feature


def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list, vocab=None):
    ## Step 1: Set parameters
    dict_size = 100
    stride = 20
    size = 20
    k = 10
    
    print(f"Parameters: dict_size={dict_size}, stride={stride}, size={size}, k={k}")
    
    ## Step 2: Build or load visual dictionary
    if vocab is None:
        print("Building visual dictionary...")
        print("Computing dense SIFT for all training images...")
        
        dense_feature_list = []
        for i, img_path in enumerate(img_train_list):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            dense_features = compute_dsift(img, stride, size)
            dense_feature_list.append(dense_features)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(img_train_list)} training images")
        
        vocab = build_visual_dictionary(dense_feature_list, dict_size)
    else:
        print("Using provided vocabulary")
    
    ## Step 3: Compute BoW features for training images
    print("Computing BoW features for training images...")
    n_train = len(img_train_list)
    feature_train = []
    
    for i, img_path in enumerate(img_train_list):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        dense_features = compute_dsift(img, stride, size)
        bow_feature = compute_bow(dense_features, vocab)
        feature_train.append(bow_feature)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_train} training images")
    
    feature_train = np.array(feature_train)
    
    ## Step 4: Convert training labels to numeric
    label_to_index = {label: idx for idx, label in enumerate(label_classes)}
    label_train = np.array([label_to_index[label] for label in label_train_list])
    
    ## Step 5: Compute BoW features for test images
    print("Computing BoW features for test images...")
    n_test = len(img_test_list)
    feature_test = []
    
    for i, img_path in enumerate(img_test_list):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        dense_features = compute_dsift(img, stride, size)
        bow_feature = compute_bow(dense_features, vocab)
        feature_test.append(bow_feature)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_test} test images")
    
    feature_test = np.array(feature_test)
    
    ## Step 6: Convert test labels to numeric
    label_test = np.array([label_to_index[label] for label in label_test_list])
    
    ## Step 7: Predict using kNN
    print("Predicting labels with kNN...")
    label_test_pred = predict_knn(feature_train, label_train, feature_test, k)
    
    ## Step 8: Build confusion matrix
    n_classes = len(label_classes)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    
    for true_label, pred_label in zip(label_test, label_test_pred):
        confusion[true_label, pred_label] += 1
    
    ## Step 9: Calculate accuracy
    correct = np.trace(confusion)
    total = np.sum(confusion)
    accuracy = correct / total
    
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    return confusion, accuracy


def predict_svm(feature_train, label_train, feature_test):
    '''
    input:

    output:

    note:
    
    '''
    n_classes = 15
    n_test = feature_test.shape[0]
    C = 10
    
    print(f"Training {n_classes} binary SVM classifiers with C={C}...")
    
    scores = np.zeros((n_test, n_classes))
    
    for class_idx in range(n_classes):
        print(f"  Training SVM for class {class_idx}...")
        
        # Create binary labels and train SVM
        binary_labels = (label_train == class_idx).astype(int)
        svm = LinearSVC(C=C, max_iter=2000, random_state=42)
        svm.fit(feature_train, binary_labels)
        
        # Get decision scores
        scores[:, class_idx] = svm.decision_function(feature_test)
    
    # Predict class with highest score
    label_test_pred = np.argmax(scores, axis=1)
    
    print("SVM prediction complete!")

    return label_test_pred


def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list, vocab=None):
    '''
    input:

    output:

    note:

    '''
    ## Step 1: Set parameters
    dict_size = 200
    stride = 20
    size = 20
    
    print(f"Parameters: dict_size={dict_size}, stride={stride}, size={size}")
    
    ## Step 2: Build or load visual dictionary
    if vocab is None:
        print("Building visual dictionary...")
        print("Computing dense SIFT for all training images...")
        
        dense_feature_list = []
        for i, img_path in enumerate(img_train_list):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            dense_features = compute_dsift(img, stride, size)
            dense_feature_list.append(dense_features)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(img_train_list)} training images")
        
        vocab = build_visual_dictionary(dense_feature_list, dict_size)
    else:
        print("Using provided vocabulary")
    
    ## Step 3: Compute BoW features for training images
    print("Computing BoW features for training images...")
    n_train = len(img_train_list)
    feature_train = []
    
    for i, img_path in enumerate(img_train_list):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        dense_features = compute_dsift(img, stride, size)
        bow_feature = compute_bow(dense_features, vocab)
        feature_train.append(bow_feature)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_train} training images")
    
    feature_train = np.array(feature_train)
    
    ## Step 4: Convert training labels to numeric
    label_to_index = {label: idx for idx, label in enumerate(label_classes)}
    label_train = np.array([label_to_index[label] for label in label_train_list])
    
    ## Step 5: Compute BoW features for test images
    print("Computing BoW features for test images...")
    n_test = len(img_test_list)
    feature_test = []
    
    for i, img_path in enumerate(img_test_list):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        dense_features = compute_dsift(img, stride, size)
        bow_feature = compute_bow(dense_features, vocab)
        feature_test.append(bow_feature)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_test} test images")
    
    feature_test = np.array(feature_test)
    
    ## Step 6: Convert test labels to numeric
    label_test = np.array([label_to_index[label] for label in label_test_list])
    
    ## Step 7: Predict using SVM (KEY DIFFERENCE!)
    print("Training SVM and predicting labels...")
    label_test_pred = predict_svm(feature_train, label_train, feature_test)
    
    ## Step 8: Build confusion matrix
    n_classes = len(label_classes)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    
    for true_label, pred_label in zip(label_test, label_test_pred):
        confusion[true_label, pred_label] += 1
    
    ## Step 9: Calculate accuracy
    correct = np.trace(confusion)
    total = np.sum(confusion)
    accuracy = correct / total
    
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

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

    ## uncommented this line to use existing vocab
    vocab = np.load("vocab.npy")
    confusion, accuracy = classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list, vocab)
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    del confusion, accuracy
    gc.collect()

    ## uncommented this line to use existing vocab
    vocab = np.load("vocab.npy")
    confusion, accuracy = classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list, vocab)
    visualize_confusion_matrix(confusion, accuracy, label_classes)
