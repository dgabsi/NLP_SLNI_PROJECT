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


def save_model(model, optimizer, models_dir, file_name, epoch_end, train_end_step, val_end_step):

    filename=os.path.join(models_dir, file_name + '.pth')


    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch_end': epoch_end,
        'train_end_step': train_end_step ,
        'val_end_step': val_end_step
    }, filename)



def load_model(models_dir,file_name):

    filename = os.path.join(models_dir,file_name)
    checkpoint = torch.load(filename)
    model_state_dict=checkpoint['model_state_dict']
    optimizer_state_dict=checkpoint['optimizer_state_dict']
    epoch_end = checkpoint['epoch_end']
    train_end_step = checkpoint['train_end_step']
    val_end_step= checkpoint['val_end_step']

    return model_state_dict, optimizer_state_dict, epoch_end, train_end_step, val_end_step


def save_to_pickle(entity, file):
    with open(file, 'wb') as file:
        pickle.dump(entity, file)
    file.close()

def load_from_pickle(file):
    with open(file, 'rb') as file:
        entity = pickle.loads(file.read())
    file.close()
    return entity

'''
def plot_all_experiments(df_results, plot_file):
    plot_file = os.path.join(models_dir, plot_file + '.png')

    run_types=df_results["run_name"].unique()
    fig = plt.figure()
    for ind, _ in enumerate(run_types):
        
        plt.plot(range(len(list_all_total_mean[ind])), list_all_total_mean[ind], label=labels[ind])
        plt.fill_between(range(len(list_all_total_mean[ind])), list_all_total_mean[ind] - list_all_total_std[ind] / 2,
                         list_all_total_mean[ind] + list_all_total_std[ind] / 2, color='red', alpha=0.2)

    plt.xlabel('Episodes')
    plt.ylabel(y_labels)
    plt.legend()
    plt.title(title)
    plt.savefig(plot_file, dpi=fig.dpi)
    plt.show()
'''

def report_result(y_predicted,y_true,model_name,labels, title):
    """"
    Print a classification report and a confusion matrix
    """
    print("{} classifer".format(model_name))
    print("Accuracy: {:.3f}".format(accuracy_score(y_true, y_predicted)))
    print(classification_report(y_true, y_predicted, target_names=labels))
    cm = confusion_matrix(y_true, y_predicted, labels=np.arange(len(labels)))
    print("Confusion matrix:\n{}".format(cm))
    img = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    figure, axes = plt.subplots(figsize=(5, 5))
    img.plot(ax=axes, xticks_rotation='vertical')
    axes.set_title(title)
    plt.show()

def plot_experiments(plot_file_path, df_results, title=''):
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    plot_file = plot_file_path
    for run_name in pd.unique(df_results["run_name"]):
        accuracy_data = df_results.loc[df_results["run_name"] == run_name, ["epochs", "val_accuracy"]].sort_values(by=["epochs"])

        axes[0].plot(accuracy_data["epochs"].to_numpy(), accuracy_data["val_accuracy"].to_numpy(), marker='o',  linestyle='solid', label=run_name)
        val_loss = df_results.loc[ df_results["run_name"] == run_name, ["epochs", "val_loss"]].sort_values(by=["epochs"])
        axes[1].plot(val_loss["epochs"].to_numpy(), val_loss["val_loss"].to_numpy(), marker='o', linestyle='solid',  label=run_name)
        train_loss = df_results.loc[ df_results["run_name"] == run_name, ["epochs", "train_loss"]].sort_values(by=["epochs"])
        axes[2].plot(train_loss["epochs"].to_numpy(), train_loss["train_loss"].to_numpy(), marker='o',
                     linestyle='solid', label=run_name)
        print(run_name)

    axes[0].set_xlabel("Epochs")
    axes[1].set_xlabel("Epochs")
    axes[2].set_xlabel("Epochs")
    axes[0].set_ylabel("val_accuracy")
    axes[1].set_ylabel("val_loss")
    axes[2].set_ylabel("train_loss")
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    axes[0].set_title(title+'Validation accuracy of experiments')
    axes[1].set_title(title+'Validation loss of experiments')
    axes[2].set_title(title+'Train loss of experiments')
    plt.show()
    fig.savefig(plot_file_path, dpi=fig.dpi)

def plot_train_val_loss(saved_dir, df_results):

    for run_name in pd.unique(df_results["run_name"]):
        fig, axes = plt.subplots(figsize=(10,10))
        model_name=pd.unique(df_results["model"])[0]
        plot_file = os.path.join(saved_dir, run_name +'_train_val_loss'+'.png')
        val_loss=df_results.loc[df_results["run_name"]==run_name,["epochs", "val_loss"]].sort_values(by=["epochs"])
        axes.plot(val_loss["epochs"], val_loss["val_loss"], marker='o', linestyle='solid', label='val_loss')
        train_loss=df_results.loc[df_results["run_name"]==run_name,["epochs", "train_loss"]].sort_values(by=["epochs"])
        axes.plot(train_loss["epochs"], train_loss["train_loss"], marker='o', linestyle='solid', label='train_loss')
        axes.set_xlabel("Epochs")
        axes.set_ylabel("loss")
        axes.legend()
        axes.set_title('Training and validation loss_'+model_name+'_'+run_name)
        plt.show()
        fig.savefig(plot_file, dpi=fig.dpi)