import argparse
import math
import os
import json
import torch
import cv2
import torchvision.transforms as transforms
import string
from train import resume

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_exp_id', type=str, default='BSAP_1')
    parser.add_argument('--output_dir', type=str, default='./model_save')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    bert_nspm, tokenizer = resume(args)
    bert_nspm = bert_nspm.to(device)

    data_path = '../raw_data'
    # test_set_path = '/content/drive/MyDrive/test_set_sequences.json'
    test_set_path = '/content/drive/MyDrive/test_set_sequences_short.json'
    with open(test_set_path, 'r') as t_f:
      test_set = json.load(t_f)
    app_list = test_set.keys()
    # app_list = ['com.urbandroid.sleep']
    total_sequences = 0
    complete_predict = 0
    image_size = 7
    for app_name in app_list:
      print(app_name)
      app_path = os.path.join(data_path, app_name)
      sequence_list = test_set[app_name]
      for sequence_name in sequence_list:
        print(sequence_name)
        sequence_path = os.path.join(app_path, sequence_name)
        clickable_compo_path = os.path.join(sequence_path, 'clickable_compo')
        metadata_path = os.path.join(sequence_path, 'metadata.json')
        groundtruth_path = os.path.join(sequence_path, 'explicit_sequence_include_scroll.json')
        # UIs_names = os.listdir(clickable_compo_path)
        task_path = os.path.join(sequence_path,'task.txt')
        # task_path = os.path.join(sequence_path,'task_short.txt')
        with open(task_path, 'r') as t_f:
          target = t_f.readline()
        print("Task: ",target)

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        try:
          with open(groundtruth_path, 'r') as f:
            raw_groundtruth = json.load(f)
        except FileNotFoundError:
            print("File Not Found Error!")
            continue

        groundtruth = []
        for groundtruth_compo in raw_groundtruth:
            if groundtruth_compo == [] or groundtruth_compo == ['scroll']:
                groundtruth.append(groundtruth_compo)
            elif len(groundtruth_compo) == 1 and groundtruth_compo[0] != [] and not isinstance(groundtruth_compo[0][0], list):
                groundtruth.append([groundtruth_compo[0][0]])
            else:
                text = ''
                for phrase in groundtruth_compo:
                    for word in phrase:
                        text = text + word[0] + ' '
                    text = text[:-1]
                    text = text + ';'
                text = text[:-1]
                groundtruth.append(text)
        UIs_names = metadata['views']

        clickable_compos_list = []
        idx = 0
        predict = True
        while idx != len(UIs_names) and predict:
            predict = False
            if not groundtruth[idx]:
                idx = idx + 1
                predict = True
                continue
            UI_name_jpg = UIs_names[idx]
            UI_name = UI_name_jpg.split('.')[0]
            UI_path = os.path.join(clickable_compo_path, UI_name)
            clips_path = os.path.join(UI_path, 'clips')
            json_path = os.path.join(UI_path, 'clickable_compos.json')
            with open(json_path, 'r') as f:
                clickable_compos_list = clickable_compos_list + json.load(f)

            if groundtruth[idx] == ['scroll']:
                idx = idx + 1
                predict = True
                continue
            most_matchable_compo = {'text': None, 'possibility': -math.inf}
            for clickable_compo in clickable_compos_list:
                compo_text = clickable_compo[0]
                compo_ner = clickable_compo[1]

                Is_img = False
                img_list = torch.zeros((1, 64, 3, image_size, image_size))
                ner_onehot_embedding = torch.zeros((64, 5))
                temp_ner_embedding = torch.zeros((1, 5))
                if compo_ner == 'CLIP_IMAGE':
                    Is_img = True

                    # turn absolute path
                    img_path = compo_text
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (image_size, image_size))
                    trans = transforms.ToTensor()
                    img = trans(img)
                    compo_text = '0'
                    temp_ner_embedding = temp_ner_embedding + torch.tensor([0, 0, 0, 0, 1])
                else:
                    for ner in compo_ner.split(';'):
                        if 'TIME' in ner or 'DATE' in ner or 'DURATION' in ner:
                            temp_ner_embedding = temp_ner_embedding + torch.tensor([0, 0, 0, 1, 0])
                        if 'PERSON' in ner:
                            temp_ner_embedding = temp_ner_embedding + torch.tensor([0, 0, 1, 0, 0])
                        if 'LOCATION' in ner or 'CITY' in ner or 'STATE_OR_PROVINCE' in ner or 'COUNTRY' in ner:
                            temp_ner_embedding = temp_ner_embedding + torch.tensor([0, 1, 0, 0, 0])
                        if 'EMAIL' in ner or 'URL' in ner:
                            temp_ner_embedding = temp_ner_embedding + torch.tensor([1, 0, 0, 0, 0])
                tokenize_output = tokenizer(compo_text, target, add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                            truncation=True,
                                            max_length=64,  # Pad & truncate all sentences.
                                            pad_to_max_length=True,
                                            return_attention_mask=True,
                                            return_tensors='pt')

                input_ids = tokenize_output['input_ids']
                SEP_index = (input_ids == 102).nonzero()[:, 1]

                ner_onehot_embedding[0:SEP_index[0]] = temp_ner_embedding
                if Is_img == True:
                    img_list[0, SEP_index[0] - 1, :] = img.reshape((3, image_size, image_size))
                attention_mask = tokenize_output['attention_mask']
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                img_list = img_list.to(device)
                ner_onehot_embedding = ner_onehot_embedding.to(device)
                model_output = bert_nspm(input_ids, attention_mask, img_list, compo_ner, ner_onehot_embedding)
                if model_output[0][0][1] > most_matchable_compo['possibility']:
                    most_matchable_compo['text'] = clickable_compo[0]
                    most_matchable_compo['possibility'] = model_output[0][0][1]
            clickable_compos_list = []
            for i in range(idx,len(groundtruth)):
                if groundtruth[i] == []:
                  continue
                if isinstance(groundtruth[i], list):
                    true_img_path = groundtruth[i][0]
                    if os.path.exists(true_img_path):
                        true_img = cv2.imread(true_img_path)
                    else:
                        true_img = None
                    pred_img_path = most_matchable_compo['text']
                    if pred_img_path is not None and os.path.exists(pred_img_path):
                        pred_img = cv2.imread(pred_img_path)
                    else:
                        pred_img = None
                    if true_img is not None and pred_img is not None:
                        true_img_resize = cv2.resize(true_img, (image_size, image_size))
                        pred_img_resize = cv2.resize(pred_img, (image_size, image_size))
                        if (true_img_resize == pred_img_resize).all():
                            # Need to modify!!!!
                            idx = i + 1
                            predict = True
                else:
                    ignore_punc_pred = most_matchable_compo['text'].translate(
                        str.maketrans('', '', string.punctuation)).translate(str.maketrans('', '', ' '))
                    ignore_punc_ground = groundtruth[i].translate(str.maketrans('', '', string.punctuation)).translate(
                        str.maketrans('', '', ' '))
                    if ignore_punc_pred == ignore_punc_ground:
                        idx = i + 1
                        predict = True
        if predict:
            complete_predict = complete_predict + 1
        total_sequences = total_sequences + 1

    if total_sequences != 0:
      complete_predict_accuracy = complete_predict / total_sequences
    print('Completion Accuracy:', complete_predict_accuracy)
