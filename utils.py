import numpy
import torch
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import preprocessing
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def save_model(model, optimizer, models_dir, file_name, epoch_start, epoch_end, train_end_step, val_end_step):

    filename=os.path.join(models_dir, file_name + '.pth')
    torch.save(model.state_dict(), filename)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch_start': epoch_start,
        'epoch_end': epoch_end,
        'train_end_step': train_end_step ,
        'val_end_step': val_end_step
    }, filename)



def load_model(models_dir,file_name):

    filename = os.path.join(models_dir,file_name)
    checkpoint = torch.load(filename)
    model_state_dict=checkpoint['model_state_dict']
    optimizer_state_dict=checkpoint['optimizer_state_dict']
    epoch_start = checkpoint['epoch_start']
    epoch_end = checkpoint['epoch_end']
    train_end_step = checkpoint['train_end_step']
    val_end_step= checkpoint['val_end_step']

    return model_state_dict, optimizer_state_dict, epoch_start, epoch_end, train_end_step, val_end_step


def save_to_pickle(entity, file):
    with open(file, 'wb') as file:
        pickle.dump(entity, file)
    file.close()

def load_from_pickle(file):
    with open(file, 'rb') as file:
        entity = pickle.loads(file.read())
    file.close()
    return entity

def report_result(y_predicted,y_true,model_name,labels, title):
    """"
    Print a classification report and a confusion matrix
    """
    print("{} classifer".format(model_name))
    print("Accuracy: {:.3f}".format(accuracy_score(y_true, y_predicted)))
    print(classification_report(y_true, y_predicted, labels=labels))
    cm = confusion_matrix(y_true, y_predicted, labels=labels)
    print("Confusion matrix:\n{}".format(cm))
    img = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    figure, axes = plt.subplots(figsize=(10, 10))
    img.plot(ax=axes, xticks_rotation='vertical')
    axes.set_title(title)
    plt.show()