#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 23:04:59 2018

@author: michal
"""

# 2.Visualization libraries
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

sns.set()

def plot_sample_img(data,output,names,title="sample",fig_size =(30,30),title_font = 50, data_num = 6,n_rows = 2,n_cols = 3, save_fig = False):
    
    """
    Description : Plot sample images from dataset
    
    Input : 
        * data -- picture dataset
        * title -- title of figure
        * title_font -- title of figure font
        * data_num -- number of pictures to plot
        * n_rows -- number of rows
        * n_cols -- number of columns
        
    """
    
    plt.figure(1,figsize = fig_size)
    for i in range(1,int((data_num+1))):
        plt.subplot(n_rows,n_cols,i)
        a = (np.asscalar(output[i]))
        plt.title(names[a],fontsize = 35)
        plt.imshow(data[i])
      
    
    plt.suptitle(title,fontsize=title_font)
    if save_fig == True:
        plt.savefig(title)
    
    del i
    return 0


def plot_training(history,val = True,save_fig = False):
    
    acc = history.history["acc"]
    loss = history.history["loss"]
    val_acc = history.history["val_acc"]
    val_loss = history.history["val_loss"]
    
    epochs = range(len(acc))
    
    plt.figure((2),figsize=(20,10))
    
    plt.subplot(2,1,1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
    
    if save_fig == True:
        plt.savefig("learning")
        
def plot_confusion_matrix(model,X_test,y_test,class_names):
    
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred,axis = 1)
    y_pred = model.predict_classes(X_test)
    cnf_matrix = confusion_matrix(np.argmax(y_test,axis = 1),y_pred)
    plt.figure(3,figsize=(14,14))
    plt.title("Confusion Matrix")
    sns.heatmap(cnf_matrix,annot=True, xticklabels = class_names, yticklabels = class_names, fmt='.0f', square = True,robust=True,
            cmap = "Blues", linewidths=4, linecolor='white')
    
    plt.figure(4,figsize=(14,14))
    plt.title("Normalized Confusion Matrix")
    cnf_matrix_normalized = cnf_matrix/cnf_matrix.sum(axis=0)
    sns.heatmap(cnf_matrix_normalized,annot=True, xticklabels = class_names, yticklabels = class_names, fmt='.2f', square = True,robust=True,
            cmap = "Blues", linewidth=4, linecolor='white')
    
   
    
    

    
    