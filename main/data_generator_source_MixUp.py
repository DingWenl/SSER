from random import sample
import random
import numpy as np
# import keras
from keras.utils import np_utils
# get the training sampels
def train_datagenerator(batchsize,train_data1,train_data2,train_data3,win_train,train_list, channel):
    while True:
        x_train, y_train = list(range(batchsize)), list(range(batchsize))
        
        target_list = list(range(40))
        sub_list = list(range(len(train_data1)))
        # parameter alpha (beta distribution coefficient) of MixUp
        alpha = 0.5
        # creating the list to save mix ratio
        index_list = list(range(batchsize))
        # the j-th smaple list index
        list_batchsize = list(range(batchsize))
        random.shuffle(list_batchsize)
        # get training samples of batchsize trials
        for i in range(int(batchsize)):
            #save mix ratio drawn from beta distribution
            index_list[i] = np.random.beta(alpha,alpha)
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
            # concatenate the four sub-input into one input
            x_s = np.concatenate((x_21, x_22, x_23), axis=-1)
            
            x_train[i] = x_s
            y_train[i] = np_utils.to_categorical(m, num_classes=40, dtype='float32')

        # get the i-th sample list
        x_train = np.array(x_train)
        # get the j-th sample list
        x_train1 = x_train[list_batchsize,:,:,:]
        # get the i-th label list
        y_train = np.reshape(y_train,(batchsize,40))
        # get the j-th label list
        y_train1 = y_train[list_batchsize,:]
        # reshape the mix ratio list
        index_x = np.reshape(index_list,(batchsize,1,1,1))
        index_y = np.reshape(index_list,(batchsize,1))
        # obtain the mixed samples and the corresponding labels
        x_mix = index_x*x_train + (1-index_x)*x_train1
        y_mix = index_y*y_train + (1-index_y)*y_train1

        
        yield x_mix, y_mix

# # get the validation samples
# def val_datagenerator(batchsize,train_data1,train_data2,train_data3,win_train,val_list, channel):
#     while True:
#         x_train1, x_train2, x_train3, y_train = list(range(batchsize)), list(range(batchsize)), list(range(batchsize)), list(range(batchsize))
#         target_list = list(range(40))
#         sub_list = list(range(len(train_data1)))
#         # get training samples of batchsize trials
#         for i in range(batchsize):
#             s = sample(sub_list, 1)[0]
#             k = sample(val_list, 1)[0]
#             m = sample(target_list, 1)[0]
#             # randomly selecting a single-sample in the single-trial, 35 is the frames of the delay-time
#             time_start = random.randint(35+125,int(1250+35+125-win_train))
#             time_end = time_start + win_train
#             # get four sub-inputs
#             x_11 = train_data1[s][k][m][:,time_start:time_end]
#             x_21 = np.reshape(x_11,(channel, win_train, 1))
#             x_train1[i]=x_21
            
#             x_12 = train_data2[s][k][m][:,time_start:time_end]
#             x_22 = np.reshape(x_12,(channel, win_train, 1))
#             x_train2[i]=x_22

#             x_13 = train_data3[s][k][m][:,time_start:time_end]
#             x_23 = np.reshape(x_13,(channel, win_train, 1))
#             x_train3[i]=x_23

#             y_train[i] = np_utils.to_categorical(m, num_classes=40, dtype='float32')
            
#         x_train1 = np.reshape(x_train1,(batchsize,channel, win_train, 1))
#         x_train2 = np.reshape(x_train2,(batchsize,channel, win_train, 1))
#         x_train3 = np.reshape(x_train3,(batchsize,channel, win_train, 1))

#         # concatenate the four sub-input into one input to make it can be as the input of the FB-tCNN's network
#         x_train = np.concatenate((x_train1, x_train2, x_train3), axis=-1)
#         y_train = np.reshape(y_train,(batchsize,40))
        
#         yield x_train, y_train


