import os
import time
import torch
import copy
import torchvision.transforms as transforms
import time
import datetime
import numpy as np
from transformers import BertTokenizer, BertModel
import transformers
import json
transformers.logging.set_verbosity_error()
from model import CLIP

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
        Validation_Loss = 0

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            former_batch = batch[0]
            latter_batch = batch[1]
            former_input_ids = former_batch['input_ids'].squeeze().to(device)
            latter_input_ids = latter_batch['input_ids'].squeeze().to(device)
            former_attention_masks = former_batch['attention_masks'].squeeze().to(device)
            latter_attention_masks = latter_batch['attention_masks'].squeeze().to(device)
            former_images = former_batch['images'].to(device)
            latter_images = latter_batch['images'].to(device)
            former_ners_embeddings = former_batch['ners_embeddings'].to(device)
            latter_ners_embeddings = latter_batch['ners_embeddings'].to(device)
            model.zero_grad()
            # outputs = model(former_input_ids, latter_input_ids, former_attention_masks, latter_attention_masks, former_images, latter_images, former_ners_embeddings, latter_ners_embeddings)
            outputs_T, outputs_I, outputs_N = model(former_input_ids, latter_input_ids, former_attention_masks, latter_attention_masks, former_images, latter_images, former_ners_embeddings, latter_ners_embeddings)
            labels = torch.arange(len(former_input_ids)).to(device)
            # loss_f = criterion(outputs, labels)
            # loss_l = criterion(outputs.T, labels)
            loss_T_f = criterion(outputs_T, labels)
            loss_T_l = criterion(outputs_T.T, labels)
            loss_I_f = criterion(outputs_I, labels)
            loss_I_l = criterion(outputs_I.T, labels)
            loss_N_f = criterion(outputs_N, labels)
            loss_N_l = criterion(outputs_N.T, labels)

            # loss = (loss_f + loss_l) / 2
            loss_T = (loss_T_f + loss_T_l) / 2
            loss_I = (loss_I_f + loss_I_l) / 2
            loss_N = (loss_N_f + loss_N_l) / 2
            loss = (loss_T + loss_I + loss_N) / 3
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

        model_id_path = os.path.join(args.output_dir, args.exp_id)
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
            former_batch = batch[0]
            latter_batch = batch[1]

            former_input_ids = former_batch['input_ids'].squeeze().to(device)
            latter_input_ids = latter_batch['input_ids'].squeeze().to(device)
            former_attention_masks = former_batch['attention_masks'].squeeze().to(device)
            latter_attention_masks = latter_batch['attention_masks'].squeeze().to(device)
            former_images = former_batch['images'].to(device)
            latter_images = latter_batch['images'].to(device)
            former_ners_embeddings = former_batch['ners_embeddings'].to(device)
            latter_ners_embeddings = latter_batch['ners_embeddings'].to(device)



            with torch.no_grad():
                # outputs = model(former_input_ids, latter_input_ids, former_attention_masks, latter_attention_masks,
                #                 former_images, latter_images, former_ners_embeddings, latter_ners_embeddings)
                outputs_T, outputs_I, outputs_N = model(former_input_ids, latter_input_ids, former_attention_masks, latter_attention_masks,
                                 former_images, latter_images, former_ners_embeddings, latter_ners_embeddings)
                labels = torch.arange(len(former_input_ids)).to(device)
                # loss_f = criterion(outputs, labels)
                # loss_l = criterion(outputs.T, labels)
                loss_T_f = criterion(outputs_T, labels)
                loss_T_l = criterion(outputs_T.T, labels)
                loss_I_f = criterion(outputs_I, labels)
                loss_I_l = criterion(outputs_I.T, labels)
                loss_N_f = criterion(outputs_N, labels)
                loss_N_l = criterion(outputs_N.T, labels)

                loss_T = (loss_T_f + loss_T_l) / 2
                loss_I = (loss_I_f + loss_I_l) / 2
                loss_N = (loss_N_f + loss_N_l) / 2
                loss = (loss_T + loss_I + loss_N) / 3
                # loss = (loss_f + loss_l) / 2

            # Accumulate the validation loss.
            total_eval_loss += loss.item()
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        Validation_Loss = avg_val_loss
        validation_loss_path = os.path.join(model_id_path, 'validation_loss')
        if not os.path.exists(validation_loss_path):
          os.mkdir(validation_loss_path)
        validation_loss_epoch_save_path = os.path.join(validation_loss_path, 'validation_loss_epoch_{}.json'.format(epoch_i))
        with open(validation_loss_epoch_save_path, 'w') as l_f:
          json.dump(Validation_Loss, l_f)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    # save models

    output_dir = os.path.join(args.output_dir, args.exp_id)
    former_bert_output_dir = os.path.join(output_dir, 'former_bert')
    latter_bert_output_dir = os.path.join(output_dir, 'latter_bert')
    former_word_embedding_dir = os.path.join(output_dir, 'former_word')
    latter_word_embedding_dir = os.path.join(output_dir, 'latter_word')
    img_embedding_dir = os.path.join(output_dir, 'img')
    ner_embedding_dir = os.path.join(output_dir, 'ner')
    former_dimension_reduction_dir = os.path.join(output_dir, 'former_dimension_reduction')
    latter_dimension_reduction_dir = os.path.join(output_dir, 'latter_dimension_reduction')


    print("Saving model to %s" % output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.former_bert_base.save_pretrained(former_bert_output_dir)
    model_to_save.latter_bert_base.save_pretrained(latter_bert_output_dir)
    tokenizer.save_pretrained(former_bert_output_dir)
    torch.save(model_to_save.former_encoder.word_embedding_layer.state_dict(), former_word_embedding_dir)
    torch.save(model_to_save.latter_encoder.word_embedding_layer.state_dict(), latter_word_embedding_dir)
    torch.save(model_to_save.image_encoder.state_dict(), img_embedding_dir)
    torch.save(model_to_save.ner_encoder.state_dict(), ner_embedding_dir)
    torch.save(model_to_save.former_encoder.dimension_reduction.state_dict(), former_dimension_reduction_dir)
    torch.save(model_to_save.latter_encoder.dimension_reduction.state_dict(), latter_dimension_reduction_dir)


def resume(args):
    model_dir = os.path.join(args.output_dir, args.exp_id)
    former_bert_output_dir = os.path.join(model_dir, 'former_bert')
    latter_bert_output_dir = os.path.join(model_dir, 'latter_bert')
    former_word_embedding_dir = os.path.join(model_dir, 'former_word')
    latter_word_embedding_dir = os.path.join(model_dir, 'latter_word')
    img_embedding_dir = os.path.join(model_dir, 'img')
    ner_embedding_dir = os.path.join(model_dir, 'ner')
    former_dimension_reduction_dir = os.path.join(model_dir, 'former_dimension_reduction')
    latter_dimension_reduction_dir = os.path.join(model_dir, 'latter_dimension_reduction')
    assert os.path.exists(former_bert_output_dir), ('Model path do not exits for %s' % former_bert_output_dir)
    assert os.path.exists(latter_bert_output_dir), ('Model path do not exits for %s' % latter_bert_output_dir)
    assert os.path.exists(former_word_embedding_dir), ('Model path do not exits for %s' % former_word_embedding_dir)
    assert os.path.exists(latter_word_embedding_dir), ('Model path do not exits for %s' % latter_word_embedding_dir)
    assert os.path.exists(img_embedding_dir), ('Model path do not exits for %s' % img_embedding_dir)
    assert os.path.exists(ner_embedding_dir), ('Model path do not exits for %s' % ner_embedding_dir)
    assert os.path.exists(former_dimension_reduction_dir), ('Model path do not exits for %s' % former_dimension_reduction_dir)
    assert os.path.exists(latter_dimension_reduction_dir), ('Model path do not exits for %s' % latter_dimension_reduction_dir)

    resumed_former_bert = BertModel.from_pretrained(former_bert_output_dir)
    resumed_tokenizer = BertTokenizer.from_pretrained(former_bert_output_dir)
    resumed_latter_bert = BertModel.from_pretrained(latter_bert_output_dir)

    resumed_former_word_embedding = torch.load(former_word_embedding_dir)
    resumed_latter_word_embedding = torch.load(latter_word_embedding_dir)
    resumed_img_embedding = torch.load(img_embedding_dir)
    resumed_ner_embedding = torch.load(ner_embedding_dir)
    resumed_former_dimension_reduction = torch.load(former_dimension_reduction_dir)
    resumed_latter_dimension_reduction = torch.load(latter_dimension_reduction_dir)

    resumed_clip = CLIP(resumed_former_bert, resumed_latter_bert, args.temperature)
    resumed_clip.image_encoder.load_state_dict(resumed_img_embedding)
    resumed_clip.former_encoder.word_embedding_layer.load_state_dict(resumed_former_word_embedding)
    resumed_clip.latter_encoder.word_embedding_layer.load_state_dict(resumed_latter_word_embedding)
    resumed_clip.former_encoder.dimension_reduction.load_state_dict(resumed_former_dimension_reduction)
    resumed_clip.latter_encoder.dimension_reduction.load_state_dict(resumed_latter_dimension_reduction)
    resumed_clip.ner_encoder.load_state_dict(resumed_ner_embedding)
    # Load a trained model and vocabulary that you have fine-tuned

    print('Resume completed for the model\n')

    return resumed_clip, resumed_tokenizer