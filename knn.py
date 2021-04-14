import scipy.io
from scipy.fft import fft
import pandas as pd
import numpy as np
import sys
import scipy.signal
from skfeature.function.similarity_based import fisher_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

np.set_printoptions(threshold=sys.maxsize)

fs=500

def bandpower(x, fmin, fmax):

    Pxx = fft(x)
    
    ind_min = fmin*3 
    ind_max = fmax*3

    Pxx_abs = np.abs(Pxx)
    Pxx_pow = np.square(Pxx_abs)

    Pxx_sum = sum(Pxx_pow[ind_min: ind_max])

    return Pxx_sum

def segmentation(x, time_length, time_shift):

    length = time_length * 500
    shift_num = time_shift * 500

    epoches_array = []

    index = 0

    while True:

        if index*shift_num+length >= len(x):
            break
        else:

            array = x[index*shift_num:index*shift_num+length]
            epoches_array.append(array)

            index += 1

    #print(len(epoches_array))

    return epoches_array

def compute(x):

    band_array = []
    rp_ratio_array = []

    for channel in range(0, 30):

        channelData = np.asarray(x[channel])

        epoches_array = segmentation(channelData, 3, 1)

        delta_array, theta_array, alpha_array, beta_array, gamma_array, total_power_array = [], [], [], [], [], []

        for epoch in epoches_array:

            delta = bandpower(epoch, 1, 4)
            theta = bandpower(epoch, 4, 8)
            alpha = bandpower(epoch, 8, 13)
            beta = bandpower(epoch, 13, 30)
            gamma = bandpower(epoch, 30, 45)

            total_power = bandpower(epoch, 1, 45)

            delta_array.append(delta)
            theta_array.append(theta)
            alpha_array.append(alpha)
            beta_array.append(beta)
            gamma_array.append(gamma)

            total_power_array.append(total_power)

        delta_avg = np.mean(delta_array)
        theta_avg = np.mean(theta_array)
        alpha_avg = np.mean(alpha_array)
        beta_avg = np.mean(beta_array)
        gamma_avg = np.mean(gamma_array)

        total_power_avg = np.mean(total_power_array)

        rp_ratio_array.append([delta_avg/total_power_avg, theta_avg/total_power_avg, alpha_avg/total_power_avg, beta_avg/total_power_avg, gamma_avg/total_power_avg])
        band_array.append([delta_avg, theta_avg, alpha_avg, beta_avg, gamma_avg])

    band_array = np.asarray(band_array)
    rp_ratio_array = np.asarray(rp_ratio_array)

    band_DF = pd.DataFrame(band_array)
    print(band_DF)

    #ratio_DF = pd.DataFrame(rp_ratio_array)

    rp_features = relative_power_lab(rp_ratio_array)

    return rp_features

def relative_power_lab(rp_ratio_array):
    
    rp_lab_array = []

    for band_index in range(0, 5):
        for channel in range(0, 30):
            for rec_channel in range(channel, 30):

                if rec_channel == channel:
                    continue
                else:
    
                    b_1 = rp_ratio_array[channel][band_index]
                    b_2 = rp_ratio_array[rec_channel][band_index]

                    rp_lab = (b_2 - b_1) / (b_1 + b_2)

                    rp_lab_array.append(rp_lab)

    rp_lab_array = np.asarray(rp_lab_array)

    return rp_lab_array

def lda(X_train, y_train, X_test, y_test):

    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, y_train)

    prediction = clf.predict(X_test)

    cal_cr_balance_cr(prediction, y_test)

def knn(X_train, y_train, X_test, y_test):

    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)

    prediction = classifier.predict(X_test)

    cal_cr_balance_cr(prediction, y_test)

def cal_cr_balance_cr(prediction, y_test):

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for index, (pred, y) in enumerate(zip(prediction, y_test)):

        if index < 10:
            if pred == y:
                TP += 1
            else:
                TN += 1
        else:
            if pred == y:
                FP += 1
            else:
                FN += 1

    CR = (TP+TN)/prediction.shape[0]

    TPR = TP/(TP + FN)
    TNR = TN/(FP + TN)

    balance_CR = (TPR + TNR)/2

    print(CR)
    print(balance_CR)

def main():
    
    mat = scipy.io.loadmat('Tee_170321.mat')

    data_raw = mat['data']

    subjects_features = []
    subjects_labels = []

    train_X = []
    train_y = []
    test_X = []
    test_y = []

    for index, patient_subjects in enumerate(data_raw[0:23]):

        patient_features = compute(patient_subjects[0])

        subjects_features.append(patient_features)
        subjects_labels.append("1")

    for index, HC_subjects in enumerate(data_raw[23:47]):

        HC_features = compute(HC_subjects[0])

        subjects_features.append(HC_features)
        subjects_labels.append("0")

    subjects_features = np.asarray(subjects_features)
    subjects_labels = np.asarray(subjects_labels)

    print(subjects_features.shape)
    print(subjects_labels.shape)

    fs_score = fisher_score.fisher_score(subjects_features, subjects_labels)

    idx = fisher_score.feature_ranking(fs_score)

    print(idx)

    first_feature = idx[0]
    second_feature = idx[1]

    for index, patient_subjects in enumerate(data_raw[0:23]):
    
        patient_features = compute(patient_subjects[0])

        if index < 13:
            train_X.append([patient_features[first_feature], patient_features[second_feature]])
            train_y.append("1")
        else:
            test_X.append([patient_features[first_feature], patient_features[second_feature]])
            test_y.append("1")


    for index, HC_subjects in enumerate(data_raw[23:47]):

        HC_features = compute(HC_subjects[0])

        if index < 14:
            train_X.append([HC_features[first_feature], HC_features[second_feature]])
            train_y.append("0")
        else:
            test_X.append([HC_features[first_feature], HC_features[second_feature]])
            test_y.append("0")

    train_X = np.asarray(train_X)
    train_y = np.asarray(train_y)
    test_X = np.asarray(test_X)
    test_y = np.asarray(test_y)

    print(train_X.shape)
    print(train_y.shape)
    print(test_X.shape)
    print(test_y.shape)

    knn(train_X, train_y, test_X, test_y)
    lda(train_X, train_y, test_X, test_y)

if __name__ == '__main__':
    main()
    