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
        # parameter alpha (beta distribution coefficient) of SSER
        alpha_s = 0.1
        alpha_u = 2.0
        # creating the list to save mix ratio
        index_list_s = list(range(batchsize))
        index_list_u = list(range(batchsize))
            # get training samples of batchsize trials
        for i in range(int(batchsize)):
            #save mix ratio drawn from beta distribution
            index_list_s[i] = np.random.beta(alpha_s,alpha_s)
            index_list_u[i] = np.random.beta(alpha_u,alpha_u)

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

            # concatenate the four sub-input into one input
            x_concatenate = np.concatenate((x_21, x_22, x_23), axis=-1)
            # save the training sample and corresponding label
            x_train[i] = x_concatenate
            y_train[i] = np_utils.to_categorical(m, num_classes=40, dtype='float32')
        # reshape the mixing ratio list
        index_style_s = np.reshape(index_list_s,(batchsize,1))
        index_style_u = np.reshape(index_list_u,(batchsize,1,1))
        # obtain original training samples
        x_original = np.array(x_train)
        # create the list to save the decorrelated sample and eigenvector matrix
        U_list,S_list,Vt_list = list(range(batchsize)), list(range(batchsize)),list(range(batchsize))
        for i in range(batchsize):
            # reshape i-th training samples to facilitate the application of RCC
            x_data = np.reshape(x_original[i],(channel, win_train*3))
            # obtain the i-th spacial, energy, and temporal representations
            U,S,Vt = np.linalg.svd(x_data, full_matrices = False)
            # save the original spacial, energy, and temporal representations
            U_list[i] = U
            S_list[i] = S
            Vt_list[i] = Vt
            
        # obtain the random selected S_j (energy representation)
        S_random = np.copy(S_list)
        random.shuffle(S_random)
        S_list = np.array(S_list)
        S_random = np.array(S_random)
        # obtain the mixed energy representation
        S_mix_list = index_style_s*S_list + (1-index_style_s)*S_random
        
        # obtain the random selected U_j (spacial representation)
        U_random = np.copy(U_list)
        random.shuffle(U_random)
        U_list = np.array(U_list)
        U_random = np.array(U_random)
        # obtain the mixed spacial representation
        U_mix_list = index_style_u*U_list + (1-index_style_u)*U_random

        # create list to save the reconstructed samples
        recon_list = list(range(batchsize))
        for i in range(batchsize):
            
            # obtain the i-th mixed energy representation
            s_mix = S_mix_list[i]
            S_mix_diag = np.diag(s_mix)
            # obtain the i-th mixed spacial representation
            U_mix = U_mix_list[i]
            # obtain the i-th invariant energy representation
            V_invariant = Vt_list[i]
            # obtain the i-th reconstructed sample
            recon_x = U_mix @ S_mix_diag @ V_invariant
            # reshape the i-th reconstructed sample to be the input of the deep learning model
            recon_x = np.reshape(recon_x,(channel, win_train,3))
            # save the i-th reconstructed sample
            recon_list[i] = recon_x
        # obtain the mini-batch training samples and labels
        x_input = np.array(recon_list)
        y_input = np.reshape(y_train,(batchsize,40))
        

        yield x_input, y_input

