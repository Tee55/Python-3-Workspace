import scipy.io
from scipy.fft import fft
import pandas as pd
import numpy as np
import sys
import scipy.signal
from skfeature.function.similarity_based import fisher_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut

np.set_printoptions(threshold=sys.maxsize)

fs=500
patient = 1
HC = 0

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

        if index*shift_num+length >= 45000:
            break
        else:

            array = x[index*shift_num:index*shift_num+length]
            epoches_array.append(array)

            index += 1

    #print(len(epoches_array))

    return epoches_array

def compute(x):

    band_array = []
    band_array_unused = []
    rp_ratio_array = []

    delta_com, theta_com, alpha_com, beta_com, gamma_com = [], [], [], [], []

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
        band_array_unused.append([delta_avg, theta_avg, alpha_avg, beta_avg, gamma_avg])

        delta_com.append(delta_avg)
        theta_com.append(theta_avg)
        alpha_com.append(alpha_avg)
        beta_com.append(beta_avg)
        gamma_com.append(gamma_avg)

    band_array.extend(delta_com)
    band_array.extend(theta_com)
    band_array.extend(alpha_com)
    band_array.extend(beta_com)
    band_array.extend(gamma_com)

    band_array = np.asarray(band_array, dtype=float)
    rp_ratio_array = np.asarray(rp_ratio_array)

    #band_DF = pd.DataFrame(band_array)
    #print(band_DF)

    #ratio_DF = pd.DataFrame(rp_ratio_array)

    rp_features = band_array
    #rp_features = relative_power_lab(rp_ratio_array)

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

def lda(X_train, y_train, X_val, y_val):

    lda = LinearDiscriminantAnalysis()
    lda_object = lda.fit(X_train, y_train)

    prediction = lda.predict(X_val)

    print(prediction)

    cal_cr_balance_cr(prediction, y_val)

    #Plot train data
    for index, X in enumerate(X_train):
        if index < 23:
            plt.scatter(X[0], X[1], c='r')
        else:
            plt.scatter(X[0], X[1], c='b')

    plt.scatter(X_val[:,0], X_val[:,1], c='g')

    x1 = np.array([np.min(X_train[:,0], axis=0), np.max(X_train[:,0], axis=0)])

    #Plot line
    b, w1, w2 = lda.intercept_[0], lda.coef_[0][0], lda.coef_[0][1]
    y1 = -(b + x1*w1)/w2    
    plt.plot(x1, y1)

    plt.show()

def knn(X_train, y_train, X_val, y_val):

    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)

    prediction = classifier.predict(X_val)

    cal_cr_balance_cr(prediction, y_val)

def cal_cr_balance_cr(prediction, y_val):

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for index, (pred, y) in enumerate(zip(prediction, y_val)):

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

    print([TP, TN, FP, FN])
    print(CR)
    print(balance_CR)

def find_first_two_features(data_raw):

    train_features = []
    train_labels = []

    for index, patient_subjects in enumerate(data_raw[0:23]):
    
        features = compute(patient_subjects[0])

        if index < 13:
            train_features.append(features)
            train_labels.append(patient)

    for index, HC_subjects in enumerate(data_raw[23:47]):

        features = compute(HC_subjects[0])

        if index < 14:
            train_features.append(features)
            train_labels.append(HC)

    train_features = np.asarray(train_features)
    train_labels = np.asarray(train_labels)

    band_DF = pd.DataFrame(train_features)
    #print(band_DF)

    fs_score = fisher_score.fisher_score(train_features, train_labels)

    idx = fisher_score.feature_ranking(fs_score)

    #print(idx)

    return idx[0], idx[1]

def main():
    
    mat = scipy.io.loadmat('Tee_170321.mat')

    data_raw = mat['data']

    train_X = []
    train_y = []
    val_X = []
    val_y = []

    first_feature, second_feature = find_first_two_features(data_raw)

    for index, patient_subjects in enumerate(data_raw[0:23]):
    
        patient_features = compute(patient_subjects[0])

        if index < 13:
            train_X.append([patient_features[first_feature], patient_features[second_feature]])
            train_y.append(patient)
        else:
            val_X.append([patient_features[first_feature], patient_features[second_feature]])
            val_y.append(patient)

    for index, HC_subjects in enumerate(data_raw[23:47]):

        HC_features = compute(HC_subjects[0])

        if index < 14:
            train_X.append([HC_features[first_feature], HC_features[second_feature]])
            train_y.append(HC)
        else:
            val_X.append([HC_features[first_feature], HC_features[second_feature]])
            val_y.append(HC)

    train_X = np.asarray(train_X)
    train_y = np.asarray(train_y)
    val_X = np.asarray(val_X)
    val_y = np.asarray(val_y)

    """
    print(train_X.shape)
    print(train_y.shape)
    print(val_X.shape)
    print(val_y.shape)
    """

    knn(train_X, train_y, val_X, val_y)
    lda(train_X, train_y, val_X, val_y)

def leave_one_out():

    mat = scipy.io.loadmat('Tee_170321.mat')

    data_raw = mat['data']

    X = []
    y = []

    first_feature, second_feature = find_first_two_features(data_raw)

    for index, subjects in enumerate(data_raw):
    
        features = compute(subjects[0])

        X.append([features[first_feature], features[second_feature]])
        if index < 23:
            y.append(patient)
        else:
            y.append(HC)

    X = np.asarray(X)
    y = np.asarray(y)

    loo = LeaveOneOut()

    for train_index, test_index in loo.split(X):

        train_X, val_X = X[train_index], X[test_index]
        train_y, val_y = y[train_index], y[test_index]

        lda(train_X, train_y, val_X, val_y)
        #knn(train_X, train_y, val_X, val_y)

if __name__ == '__main__':
    #main()
    leave_one_out()
    