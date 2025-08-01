
from random import sample
import random
import numpy as np
import keras
from keras.utils import np_utils
# get the training sampels
def train_datagenerator(batchsize,train_data1,train_data2,train_data3,win_train,train_list, channel):
    while True:
        x_train, y_train = list(range(batchsize)), list(range(batchsize))
        
        target_list = list(range(40))
        sub_list = list(range(len(train_data1)))
        # parameter alpha of RCC
        alpha = 0.3
        # list containing the mixing ratios
        index_list = list(range(batchsize))
        # each mini-batch has a probability of 0.5 to determine whether to apply RCC during training 
        if_RCC_list = [0,1]
        RCC_index = sample(if_RCC_list, 1)[0]
        # RCC is applied to this mini-batch
        if RCC_index == 0:
            # get training samples of batchsize trials
            for i in range(int(batchsize)):
                index_list[i] = np.random.beta(alpha,alpha)

                m = sample(target_list, 1)[0]
                k = sample(train_list, 1)[0]
                s = sample(sub_list, 1)[0]
                # randomly selecting a single-sample in the single-trial, 35 is the frames of the delay-time
                time_start = random.randint(35+125,int(1250+35+125-win_train))
                time_end = time_start + win_train
                # get three sub-inputs
                x_11 = train_data1[s][k][m][:,time_start:time_end]
                x_21 = np.reshape(x_11,(channel, win_train, 1))

                x_12 = train_data2[s][k][m][:,time_start:time_end]
                x_22 = np.reshape(x_12,(channel, win_train, 1))

                x_13 = train_data3[s][k][m][:,time_start:time_end]
                x_23 = np.reshape(x_13,(channel, win_train, 1))

                # concatenate the four sub-input into one input to make it can be as the input of the FB-tCNN's network
                x_concatenate = np.concatenate((x_21, x_22, x_23), axis=-1)
                # save the training sample and corresponding label
                x_train[i] = x_concatenate
                y_train[i] = np_utils.to_categorical(m, num_classes=40, dtype='float32')
            # reshape the mixing ration list
            index_style = np.reshape(index_list,(batchsize,1,1))
            # obtain original training samples
            x_original = np.array(x_train)
            # create the list to save the decorrelated sample and eigenvector matrix
            featVec_list,x_decorrelated_list = list(range(batchsize)), list(range(batchsize))
            for i in range(batchsize):
                # reshape i-th training samples to facilitate the application of RCC
                x_data = np.reshape(x_original[i],(channel, win_train*3))
                # Step1, obtain the i-th covariance matrix
                x_cov = np.cov(x_data)
                # Step2, obtain the i-th eigenvector matrix
                _, featVec=np.linalg.eig(x_cov)
                # Step3, obtain the i-th decorrelated sample
                x_decorrelated = np.dot(featVec.T,x_data) #np.linalg.inv(featVec1),featVec1.T
                # save the decorrelated sample and eigenvector matrix
                x_decorrelated_list[i] = x_decorrelated
                featVec_list[i] = featVec
            # obtain the random selected U_j
            featVec_random = np.copy(featVec_list)
            random.shuffle(featVec_random)

            featVec_list = np.array(featVec_list)
            featVec_random = np.array(featVec_random)
            # Step4, obtain the mixed eigenvector matrix
            featVec_list = index_style*featVec_list + (1-index_style)*featVec_random
            # create list to save the reconstructed samples
            recon_list = list(range(batchsize))
            for i in range(batchsize):
                # Step5, obtain the i-th reconstructed sample
                recon_x = np.dot(featVec_list[i],x_decorrelated_list[i])
                # reshape the i-th reconstructed sample to be the input of the deep learnning model
                recon_x = np.reshape(recon_x,(channel, win_train,3))
                # save the i-th reconstructed sample
                recon_list[i] = recon_x
            # obtain the mini-batch training samples and labels
            x_input = np.array(recon_list)
            y_input = np.reshape(y_train,(batchsize,40))
        # RCC is not applied to this mini-batch
        else:
            # get training samples of batchsize trials
            for i in range(int(batchsize)):
                m = sample(target_list, 1)[0]
                k = sample(train_list, 1)[0]
                s = sample(sub_list, 1)[0]
                # randomly selecting a single-sample in the single-trial, 35 is the frames of the delay-time
                time_start = random.randint(35+125,int(1250+35+125-win_train))
                time_end = time_start + win_train
                # get four sub-inputs
                x_11 = train_data1[s][k][m][:,time_start:time_end]
                x_21 = np.reshape(x_11,(channel, win_train, 1))

                x_12 = train_data2[s][k][m][:,time_start:time_end]
                x_22 = np.reshape(x_12,(channel, win_train, 1))

                x_13 = train_data3[s][k][m][:,time_start:time_end]
                x_23 = np.reshape(x_13,(channel, win_train, 1))
                
                x_concatenate = np.concatenate((x_21, x_22, x_23), axis=-1)
                
                x_train[i] = x_concatenate
                y_train[i] = np_utils.to_categorical(m, num_classes=40, dtype='float32')

            x_input = np.array(x_train)
            y_input = np.reshape(y_train,(batchsize,40))

        yield x_input, y_input
