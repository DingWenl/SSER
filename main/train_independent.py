from keras.callbacks import ModelCheckpoint
from net_cnnformer import cnnformer
from data_generator_source_SSER import train_datagenerator#, val_datagenerator
import scipy.io as scio 
from scipy import signal
from keras.models import Model#,load_model
from keras.layers import Input
import numpy as np
# from random import sample
import os
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.losses import CategoricalCrossentropy
# get the filtered EEG-data, label and the start time of each trial of the dataset
def get_train_data(wn11,wn21,wn12,wn22,wn13,wn23,path):
    # read the data
    data = scio.loadmat(path)
    # get the EEG-data of the selected electrodes and downsampling it
    data_1 = data['data']
    c1 = [47,53,54,55,56,57,60,61,62]
    
    train_data = data_1[c1,:,:,:]
    # get the filtered EEG-data with six-order Butterworth filter of the first sub-filter
    block_data_list1 = []
    for i in range(train_data.shape[3]):
        target_data_list = []
        for j in range(train_data.shape[2]):
            channel_data_list = []
            for k in range(train_data.shape[0]):
                b, a = signal.butter(6, [wn11,wn21], 'bandpass')
                filtedData = signal.filtfilt(b, a, train_data[k,:,j,i])
                channel_data_list.append(filtedData)
            channel_data_list = np.array(channel_data_list)
            target_data_list.append(channel_data_list)
        block_data_list1.append(target_data_list)
    # get the filtered EEG-data with six-order Butterworth filter of the second sub-filter
    block_data_list2 = []
    for i in range(train_data.shape[3]):
        target_data_list = []
        for j in range(train_data.shape[2]):
            channel_data_list = []
            for k in range(train_data.shape[0]):
                b, a = signal.butter(6, [wn12,wn22], 'bandpass')
                filtedData = signal.filtfilt(b, a, train_data[k,:,j,i])
                channel_data_list.append(filtedData)
            channel_data_list = np.array(channel_data_list)
            target_data_list.append(channel_data_list)
        block_data_list2.append(target_data_list)

    block_data_list3 = []
    for i in range(train_data.shape[3]):
        target_data_list = []
        for j in range(train_data.shape[2]):
            channel_data_list = []
            for k in range(train_data.shape[0]):
                b, a = signal.butter(6, [wn13,wn23], 'bandpass')
                filtedData = signal.filtfilt(b, a, train_data[k,:,j,i])
                channel_data_list.append(filtedData)
            channel_data_list = np.array(channel_data_list)
            target_data_list.append(channel_data_list)
        block_data_list3.append(target_data_list)

    return block_data_list1, block_data_list2, block_data_list3

if __name__ == '__main__':
    # open the GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    #%% Setting hyper-parameters
    # ampling frequency after downsampling
    fs = 250
    # the number of the electrode channels
    channel = 9
    # the mini-batch size of the training process
    batchsize = 512
    
    # the filter ranges of the four sub-filters in the filter bank
    f_down1 = 6
    f_up1 = 50
    wn11 = 2*f_down1/fs
    wn21 = 2*f_up1/fs
    
    f_down2 = 14
    f_up2 = 50
    wn12 = 2*f_down2/fs
    wn22 = 2*f_up2/fs
    
    f_down3 = 22
    f_up3 = 50
    wn13 = 2*f_down3/fs
    wn23 = 2*f_up3/fs
    #%% Training the models of multi-subjects and multi-time-window
    # the list of the time-window
    t_train_list = [1.0]# [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]

    # selecting the training subject
    for group_n in range(5):
        data_list1,data_list2,data_list3 = [],[],[]
        total_subject_list = list(range(1,36))
        test_subject_list = list(range(group_n*7+1,(group_n+1)*7+1))
        train_subject_list = [subject_n for subject_n in total_subject_list if subject_n not in test_subject_list]
        for sub_selelct in train_subject_list:#sub_list:
            print(sub_selelct)
            # the path of the dataset and you need change it for your training
            path = '/data/dwl/ssvep/benchmark/S%d.mat'%sub_selelct
            # get the filtered EEG-data of three sub-input, label and the start time of all trials of the training data
            data1, data2, data3 = get_train_data(wn11,wn21,wn12,wn22,wn13,wn23,path)
            data_list1.append(data1)
            data_list2.append(data2)
            data_list3.append(data3)
            # selecting the training time-window
        for t_train in t_train_list:
            # transfer time to frame
            win_train = int(fs*t_train)
            # all the blocks from source subjects used to train models
            train_list = list(range(6))
            # data generator (generate the taining samples of batchsize trials)
            train_gen = train_datagenerator(batchsize,data_list1, data_list2, data_list3,win_train,train_list, channel)
            # val_gen = val_datagenerator(batchsize,data_list1, data_list2, data_list3,win_train,val_list, channel)#, t_train)
            #%% setting the network (CNN-Former)
            input_shape = (channel, win_train, 3)
            input_tensor = Input(shape=input_shape)
            # using the CNN-Former model
            preds = cnnformer(input_tensor)
            model = Model(input_tensor, preds)
            # the path of the saved model and you need to change it
            model_path = '/data/dwl/ssvep/model/benchmark_test0726/cnnformer_sser/pre_%3.1fs_02_%d.h5'%(t_train,group_n)
            model_checkpoint = ModelCheckpoint(model_path, monitor='loss',verbose=1, save_best_only=True,mode='auto')
            # learning rate = 0.01, β₁ = 0.9, β₂ = 0.999, ε = 1e-7, mini-batch size = 512, number of iterations = 40,000
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            # set the number of training epochs,iteration = steps_per_epoch * epochs = 40,000
            train_epoch = 4000
            # training
            history = model.fit_generator(
                    train_gen,
                    steps_per_epoch= 10, 
                    epochs=train_epoch,
                    validation_data=None, # 
                    validation_steps=1,
                    callbacks=[model_checkpoint]
                    )
    # # show the process of the taining
            # epochs=range(len(history.history['loss']))
            # plt.subplot(221)
            # plt.plot(epochs,history.history['accuracy'],'b',label='Training acc')
            # plt.plot(epochs,history.history['val_accuracy'],'r',label='Validation acc')
            # plt.title('Traing and Validation accuracy')
            # plt.legend()
            # # plt.savefig('D:/dwl/code_ssvep/DL/cross_session/m_coyy/photo/model_V3.1_acc1.jpg')
            
            # plt.subplot(222)
            # plt.plot(epochs,history.history['loss'],'b',label='Training loss')
            # plt.plot(epochs,history.history['val_loss'],'r',label='Validation val_loss')
            # plt.title('Traing and Validation loss')
            # plt.legend()
            # plt.savefig('/home/Dwl/result/plt/model_2.5s_loss0%d.jpg'%group_n)
            
            # plt.show()






