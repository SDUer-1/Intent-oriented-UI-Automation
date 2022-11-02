import os
import time
import torch
import copy
import torchvision.transforms as transforms
import time
import datetime
import numpy as np
import json
from transformers import BertTokenizer, BertForNextSentencePrediction

from model import BertNSPM

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = torch.nn.CrossEntropyLoss()

def train(args, model, optimizer, scheduler, tokenizer, dataloaders):
    train_dataloader, validation_dataloader = dataloaders
    epochs = args.epochs

    # training

    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()
        total_train_loss = 0
        model.train()
        Train_Loss = []
        Validation_Loss_Accuracy = {'loss': 0, 'accuracy': 0}
        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            b_input_ids = batch['input_ids'].squeeze().to(device, non_blocking=True)
            b_attention_masks = batch['attention_masks'].squeeze().to(device, non_blocking=True)
            b_labels = batch['label'].to(device, non_blocking=True)
            b_ners = batch['ners']
            b_images = batch['images'].to(device, non_blocking=True)
            b_ners_embeddings = batch['ners_embeddings'].to(device, non_blocking=True)

            model.zero_grad()
            outputs = model(b_input_ids,b_attention_masks, b_images, b_ners, b_ners_embeddings)[0]
            loss = criterion(outputs, b_labels)

            total_train_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()
            if step % 5 == 0 and not step == 0:
              Train_Loss.append(loss.item())
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)
        
        model_id_path = os.path.join(args.output_dir, args.save_exp_id)
        if not os.path.exists(model_id_path):
          os.mkdir(model_id_path)
        
        train_loss_path = os.path.join(model_id_path, 'train_loss')
        if not os.path.exists(train_loss_path):
          os.mkdir(train_loss_path)
        train_loss_epoch_save_path = os.path.join(train_loss_path, 'train_loss_epoch_{}.json'.format(epoch_i))
        with open(train_loss_epoch_save_path, 'w') as l_f:
          json.dump(Train_Loss, l_f)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            b_input_ids = batch['input_ids'].squeeze().to(device)
            b_attention_masks = batch['attention_masks'].squeeze().to(device)
            b_labels = batch['label'].to(device)
            b_ners = batch['ners']
            b_images = batch['images'].to(device)
            b_ners_embeddings = batch['ners_embeddings'].to(device)

            with torch.no_grad():
                outputs = model(b_input_ids,b_attention_masks, b_images, b_ners, b_ners_embeddings)[0]
                loss = criterion(outputs, b_labels)

            # Accumulate the validation loss.
            total_eval_loss += loss.item()
            outputs = outputs.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_accuracy += flat_accuracy(outputs, label_ids)
            
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        Validation_Loss_Accuracy['loss'] = avg_val_loss
        Validation_Loss_Accuracy['accuracy'] = avg_val_accuracy
        validation_loss_accuracy_path = os.path.join(model_id_path, 'validation_loss_accuracy')
        if not os.path.exists(validation_loss_accuracy_path):
          os.mkdir(validation_loss_accuracy_path)
        validation_loss_accuracy_epoch_save_path = os.path.join(validation_loss_accuracy_path, 'validation_loss_accuracy_epoch_{}.json'.format(epoch_i))
        with open(validation_loss_accuracy_epoch_save_path, 'w') as l_f:
          json.dump(Validation_Loss_Accuracy, l_f)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    # save models

    output_dir = os.path.join(args.output_dir, args.save_exp_id)
    bert_output_dir = os.path.join(output_dir, 'bert')
    word_embedding_dir = os.path.join(output_dir, 'word')
    img_embedding_dir = os.path.join(output_dir, 'img')
    ner_embedding_dir = os.path.join(output_dir, 'ner')

    print("Saving model to %s" % output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.bert_base.save_pretrained(bert_output_dir)
    tokenizer.save_pretrained(bert_output_dir)
    torch.save(model_to_save.word_embedding_layer.state_dict(), word_embedding_dir)
    torch.save(model_to_save.resnet_encoder.state_dict(), img_embedding_dir)
    torch.save(model_to_save.ner_embedding_layer.state_dict(), ner_embedding_dir)


def resume(args):
    model_dir = os.path.join(args.output_dir, args.resume_exp_id)
    bert_model_dir = os.path.join(model_dir, 'bert')
    word_embedding_dir = os.path.join(model_dir, 'word')
    img_embedding_dir = os.path.join(model_dir, 'img')
    ner_embedding_dir = os.path.join(model_dir, 'ner')
    assert os.path.exists(bert_model_dir), ('Model path do not exits for %s' % bert_model_dir)
    assert os.path.exists(word_embedding_dir), ('Model path do not exits for %s' % word_embedding_dir)
    assert os.path.exists(img_embedding_dir), ('Model path do not exits for %s' % img_embedding_dir)
    assert os.path.exists(ner_embedding_dir), ('Model path do not exits for %s' % ner_embedding_dir)

    resumed_bert = BertForNextSentencePrediction.from_pretrained(bert_model_dir)
    resumed_tokenizer = BertTokenizer.from_pretrained(bert_model_dir)
    resumed_word_embedding = torch.load(word_embedding_dir)
    resumed_img_embedding = torch.load(img_embedding_dir)
    resumed_ner_embedding = torch.load(ner_embedding_dir)

    bert_nspm = BertNSPM(resumed_bert)
    bert_nspm.resnet_encoder.load_state_dict(resumed_img_embedding)
    bert_nspm.word_embedding_layer.load_state_dict(resumed_word_embedding)
    bert_nspm.ner_embedding_layer.load_state_dict(resumed_ner_embedding)
    # Load a trained model and vocabulary that you have fine-tuned

    print('Resume completed for the model\n')

    return bert_nspm, resumed_tokenizer
