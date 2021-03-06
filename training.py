import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.tensorboard import SummaryWriter
import snli_dataset
import evaluation
import torch.nn.functional as F
import time
from torch.utils import data
import os
import utils
import time
import datetime
import params
import math

def train_snli(model, train_dataset,val_dataset, device, model_name, config, writer, models_dir, checkpoint_file=None):
    '''
    Train model according to configuration. save to checkping and load from is requested. output to tensorboard and history dict
    :param model:
    :param train_dataset:
    :param val_dataset:
    :param device:
    :param model_name:  string model name
    :param config: dictionary of configuration (lr for now)
    :param writer: tensorboard writer
    :param models_dir: directory of checkpoints
    :param checkpoint_file: if not None load model from this file
    :return:
    '''


    train_dataloader_args = {'batch_size': params.BATCH_SIZE, 'shuffle': False, 'collate_fn': train_dataset.collate_fun}
    train_dataloader = data.DataLoader(train_dataset, **train_dataloader_args)

    val_dataloader_args = {'batch_size': params.BATCH_SIZE, 'shuffle': False, 'collate_fn': val_dataset.collate_fun}
    val_dataloader = data.DataLoader(val_dataset, **val_dataloader_args)

    optimizer = optim.Adam(model.parameters(), **config['config_optim'])
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.99)

    criterion = nn.CrossEntropyLoss()

    history = {'Epoch': [], 'Train loss': [], 'Val loss':[], 'Val accuracy': [] , 'Train iter':[], 'Val iter':[]}

    #For Tensorboard graphs
    train_iter_start = 0
    val_iter_start = 0
    epoch_start=0

    #prev val loss and patience for stopping in case the validation loss deteriorates
    max_patience = 3
    prev_val_loss=math.inf
    patience=0

    if checkpoint_file:
        model_state, optimizer_state, epoch_start,train_iter_start, val_iter_start=utils.load_model(models_dir, checkpoint_file)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    num_iters_train=0
    num_iters_val=0

    print(f"Start Training {model_name} {config['run_name']} ")

    num_epochs=config['num_epochs']

    for epoch in range(1, num_epochs+1):
        epoch_train_loss = 0
        epoch_val_loss = 0
        val_corrects=0
        num_dataset_token=0

        for _, batch in enumerate(train_dataloader):

            #print(f"Start batch {time.time()}")
            model.train()
            #inputs=batch['inputs'].to(device)
            #labels= batch.pop('labels', None)  #batch['labels'].to(device)
            #attention_padding_mask=batch['attention_padding_mask'].to(device)
            #token_type_ids=batch['token_type_ids'].to(device)

            batch_data=batch.copy()

            for key in batch_data:
                batch_data[key]=batch_data[key].to(device)

            labels = batch_data.pop('labels', None)

            optimizer.zero_grad()

            outputs_logits=model(**batch_data) #inputs, attention_padding_mask, token_type_ids


            #pred_output_logits=pred_output_logits.transpose(1,2)
            #outputs = outputs[:, 1:]
            #print(inputs)
            #print(outputs)
            #print(pred_output)
            loss = criterion(outputs_logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            #scheduler.step() #optimizer.step()
            epoch_train_loss+=loss.item()
            num_iters_train+=1

            writer.add_scalar('Training running loss', loss.item(), num_iters_train+train_iter_start)
            if not num_iters_train%1000:
                print(
                    f"Step {num_iters_train}/{len(train_dataloader)*num_epochs} Train running loss: {loss.item():.2f}")
            #print(f"Batch{train_batch}/{epoch} Train loss: {loss.item():.2f}")

            #print(f"End batch {time.time()}")
        epoch_train_loss=epoch_train_loss/len(train_dataloader)

        for _, batch in enumerate(val_dataloader):

            #first_token_output = torch.unsqueeze(inputs[:, -1], 1)
            model.eval()

            batch_data = batch.copy()
            #output_no_padding=(outputs!=(torch.full(outputs.shape, padding_token).to(device))).to(float)[:,1:]

            #inputs = batch['inputs'].to(device)
            #labels = batch['labels'].to(device)
            #attention_padding_mask = batch['attention_padding_mask'].to(device)
            #token_type_ids = batch['token_type_ids'].to(device)

            for key in batch_data:
                batch_data[key]=batch_data[key].to(device)

            labels = batch_data.pop('labels', None)
            batch_size=labels.size(0)

            if num_iters_val == 0:
                input_to_graph=tuple([value for value in batch_data.values()])
                writer.add_graph(model, input_to_graph)

            outputs_logits=model(**batch_data)

            loss = criterion(outputs_logits, labels)
            epoch_val_loss += loss.item()
            outputs = torch.argmax(F.softmax(outputs_logits, dim=1),dim=1)
            val_batch_correct=torch.sum((outputs==labels).to(float)).item()
            val_corrects += val_batch_correct

            num_iters_val += 1


            #Tensodrboard outputs
            writer.add_scalar('Validation running loss', loss.item(), num_iters_val+val_iter_start)
            writer.add_scalar('Validation running accuracy', val_batch_correct*100/batch_size, num_iters_val+val_iter_start)


        #print(dataset.tokenizer.decode(example_inputs))
        #print(model(example_inputs,example_attention_padding_mas,example_token_type_ids))
        #print(evaluation.generate_response(model, 'sotuser I need a train from cambridge to norwich please. eotuser', vocab, device))

        epoch_val_loss = epoch_val_loss / len(val_dataloader)
        epoch_val_accuracy = (val_corrects * 100. / len(val_dataset))

        print('*********************************************************************************')
        print(f"{model_name} Epoch{epoch+epoch_start}/{num_epochs+epoch_start} Train loss: {epoch_train_loss:.2f}, Val loss: {epoch_val_loss:.2f}  Val accuracy: {epoch_val_accuracy:.2f}  Run params:{config['run_name']}")
        print('*********************************************************************************')

        writer.add_scalar('Training loss', epoch_train_loss, epoch+epoch_start)
        writer.add_scalar('Validation loss', epoch_val_loss, epoch+epoch_start)
        writer.add_scalar('Val accuracy', epoch_val_accuracy, epoch+epoch_start)

        #Appedn results to history dict
        history['Epoch'].append(epoch+epoch_start)
        history['Train loss'].append(epoch_train_loss)
        history['Val loss'].append(epoch_val_loss)
        history['Val accuracy'].append(epoch_val_accuracy)
        history['Train iter'].append(train_iter_start+num_iters_train)
        history['Val iter'].append(val_iter_start+num_iters_val)

        #Save checkpint for each epoch
        date_str = datetime.datetime.now().strftime("%m%d%Y %H")
        file_name = model_name+ config['run_name']+' ' +' epoch:'+str(epoch+epoch_start)+' train_iter:'+str(train_iter_start+num_iters_train)+' val_iter:'+str(val_iter_start+num_iters_val)+date_str
        utils.save_model(model, optimizer, models_dir, file_name, epoch+epoch_start, train_iter_start+num_iters_train, val_iter_start+num_iters_val )

        #Early stopping functionality
        if prev_val_loss<epoch_val_loss:
            patience+=1
        prev_val_loss=epoch_val_loss
        if patience>max_patience:
            print(f"Validation loss has not decreased in more than {max_patience} episodes! Stopping training! ")
            break


    return history

