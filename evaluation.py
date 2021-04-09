import torch
import torch.nn.functional as F
from torch.utils import data
import utils
import os

def evaluate_on_dataset(test_dataset, model, device):
    '''
    Evaluate a model on a testsdataset
    :param test_dataset:
    :param model:
    :param device:
    :return: predicted labels, true labels , accuracy score
    '''

    test_corrects=0
    batch_size=64

    test_dataloader_args = {'batch_size': batch_size, 'shuffle': False, 'collate_fn': test_dataset.collate_fun}
    test_dataloader = data.DataLoader(test_dataset, **test_dataloader_args)

    test_outputs=[]
    test_labels=[]

    model.to(device)

    for _, batch in enumerate(test_dataloader):

        model.eval()

        batch_data = batch.copy()

        for key in batch_data:
            batch_data[key] = batch_data[key].to(device)

        labels = batch_data.pop('labels', None)
        test_labels.extend(labels.detach().cpu().numpy())

        outputs_logits = model(**batch_data)

        outputs = torch.argmax(F.softmax(outputs_logits, dim=1), dim=1)
        test_outputs.extend(outputs.detach().cpu().numpy())
        test_batch_correct = torch.sum((outputs == labels).to(float))
        test_corrects += test_batch_correct

    test_accuracy = test_corrects * 100. / len(test_dataloader.dataset)
    print(f"Test Accuracy: {test_accuracy:.2f} % ")

    return test_outputs, test_labels, test_accuracy


def run_test_and_plot_cm(model, test_dataset, models_dir, best_checkpoint, model_name, device):
    '''
    evealuate a classification model on a test dataset and print classification report and confusion matrix
    :param model:
    :param test_dataset:
    :param models_dir:
    :param best_checkpoint:
    :param model_name:
    :param device:
    :return:
    '''

    #filename = os.path.join(models_dir, best_checkpoint)
    model_state_dict, _, _, _, _=utils.load_model(models_dir,best_checkpoint)
    model.load_state_dict(model_state_dict)

    outputs, labels, accuracy= evaluate_on_dataset(test_dataset, model, device)

    #Show classification report and confusion matrix
    title='Test Results '+model_name
    utils.report_result(outputs, labels,model_name,test_dataset.LABELS, title)

    return accuracy






