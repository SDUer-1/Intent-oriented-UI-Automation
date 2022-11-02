import argparse
import os
import json
import torch
import cv2
import torchvision.transforms as transforms
import string
import math
from train import resume

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, default='CAM_0')
    parser.add_argument('--output_dir', type=str, default='./model_save')
    parser.add_argument('--temperature', type=int, default=1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    accuracies_sample_level = []
    args = parse_args()
    cam, tokenizer = resume(args)
    cam = cam.to(device).eval()

    data_path = '../raw_data'
    # test_set_path = '/content/drive/MyDrive/test_set_sequences.json'
    test_set_path = '/content/drive/MyDrive/test_set_sequences_short.json'
    with open(test_set_path, 'r') as t_f:
      test_set = json.load(t_f)
    app_list = test_set.keys()
    # app_list = ['com.urbandroid.sleep']
    total_samples_all = 0
    correct_samples_all = 0
    for app_name in app_list:
      print(app_name)
      app_path = os.path.join(data_path, app_name)
      sequence_list = test_set[app_name]
      # sequence_list = ['f748a8a5-c1eb-494f-90e9-ef9e4f761052']
      for sequence_name in sequence_list:
        print(sequence_name)
        sequence_path = os.path.join(app_path, sequence_name)
        clickable_compo_path = os.path.join(sequence_path, 'clickable_compo')
        metadata_path = os.path.join(sequence_path, 'metadata.json')
        groundtruth_path = os.path.join(sequence_path, 'explicit_sequence_include_scroll.json')
        # UIs_names = os.listdir(clickable_compo_path)
        # task_path = os.path.join(sequence_path,'task.txt')
        task_path = os.path.join(sequence_path,'task_short.txt')
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
        total_samples = 0
        correct_samples = 0
        image_size = 64
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
        for (idx, UI_name_jpg) in enumerate(UIs_names):
            if not groundtruth[idx]:
                continue
            UI_name = UI_name_jpg.split('.')[0]
            UI_path = os.path.join(clickable_compo_path, UI_name)
            clips_path = os.path.join(UI_path, 'clips')
            json_path = os.path.join(UI_path, 'clickable_compos.json')
            with open(json_path, 'r') as f:
                clickable_compos_list = clickable_compos_list + json.load(f)

            if groundtruth[idx] == ['scroll']:
                continue

            most_matchable_compo = {'text': None, 'possibility': -math.inf}
            for clickable_compo in clickable_compos_list:
                compo_text = clickable_compo[0]
                compo_ner = clickable_compo[1]

                Is_img = False
                former_img_list = torch.zeros((1, 64, 3, image_size, image_size))
                latter_img_list = torch.zeros((1, 64, 3, image_size, image_size))
                former_ner_onehot_embedding = torch.zeros((1, 5))
                latter_ner_onehot_embedding = torch.zeros((1, 5))
                if compo_ner == 'CLIP_IMAGE':
                    Is_img = True

                    # turn absolute path
                    img_path = compo_text
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (image_size, image_size))
                    trans = transforms.ToTensor()
                    img = trans(img)
                    former_img_list[0][1, :] = img
                    compo_text = '0'
                    former_ner_onehot_embedding = former_ner_onehot_embedding + torch.tensor([0, 0, 0, 0, 1])
                else:
                    for ner in compo_ner.split(';'):
                        if 'TIME' in ner or 'DATE' in ner or 'DURATION' in ner:
                            former_ner_onehot_embedding = former_ner_onehot_embedding + torch.tensor([0, 0, 0, 1, 0])
                        if 'PERSON' in ner:
                            former_ner_onehot_embedding = former_ner_onehot_embedding + torch.tensor([0, 0, 1, 0, 0])
                        if 'LOCATION' in ner or 'CITY' in ner or 'STATE_OR_PROVINCE' in ner or 'COUNTRY' in ner:
                            former_ner_onehot_embedding = former_ner_onehot_embedding + torch.tensor([0, 1, 0, 0, 0])
                        if 'EMAIL' in ner or 'URL' in ner:
                            former_ner_onehot_embedding = former_ner_onehot_embedding + torch.tensor([1, 0, 0, 0, 0])
                former_output = tokenizer(compo_text, add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                            truncation=True,
                                            max_length=64,  # Pad & truncate all sentences.
                                            pad_to_max_length=True,
                                            return_attention_mask=True,
                                            return_tensors='pt')
                                    
                latter_output = tokenizer(target, add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                            truncation=True,
                                            max_length=64,  # Pad & truncate all sentences.
                                            pad_to_max_length=True,
                                            return_attention_mask=True,
                                            return_tensors='pt')

                former_input_ids = former_output['input_ids'].to(device)
                latter_input_ids = latter_output['input_ids'].to(device)
                former_attention_mask = former_output['attention_mask'].to(device)
                latter_attention_mask = latter_output['attention_mask'].to(device)
                former_img_list = former_img_list.to(device)
                latter_img_list = latter_img_list.to(device)
                former_ner_onehot_embedding = former_ner_onehot_embedding.to(device)
                latter_ner_onehot_embedding = latter_ner_onehot_embedding.to(device)

                model_output = cam(former_input_ids, latter_input_ids, former_attention_mask, latter_attention_mask, former_img_list, latter_img_list, former_ner_onehot_embedding, former_ner_onehot_embedding)
                if model_output.mean() > most_matchable_compo['possibility']:
                    most_matchable_compo['text'] = clickable_compo[0]
                    most_matchable_compo['possibility'] = model_output.mean()
            clickable_compos_list = []
            if isinstance(groundtruth[idx], list):
                true_img_path = groundtruth[idx][0]
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
                        correct_samples = correct_samples + 1
                        correct_samples_all = correct_samples_all + 1
            else:
                ignore_punc_pred = most_matchable_compo['text'].translate(
                    str.maketrans('', '', string.punctuation)).translate(str.maketrans('', '', ' '))
                ignore_punc_ground = groundtruth[idx].translate(str.maketrans('', '', string.punctuation)).translate(
                    str.maketrans('', '', ' '))
                if ignore_punc_pred == ignore_punc_ground:
                    correct_samples = correct_samples + 1
                    correct_samples_all = correct_samples_all + 1

            total_samples = total_samples + 1
            total_samples_all = total_samples_all + 1
        if total_samples != 0:
          accuracy_per_UI = correct_samples / total_samples
          accuracies_sample_level.append(accuracy_per_UI)
          print('Accuracy each UI:', accuracy_per_UI)
    if total_samples_all != 0:
      accuracy_all_UI = correct_samples_all / total_samples_all
      print('Accuracy on all UIs: ', accuracy_all_UI)
    model_path = os.path.join(args.output_dir, args.exp_id)
    accuracy_path = os.path.join(model_path, 'accuracy_sample_level.json')
    with open(accuracy_path, 'w') as f:
      json.dump(accuracies_sample_level, f)