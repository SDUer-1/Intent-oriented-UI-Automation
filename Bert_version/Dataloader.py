import json
import ijson
import cv2
import torch
import numpy as np
import copy
from torch.utils.data import Dataset
from transformers import BertTokenizer
import torchvision.transforms as transforms
from stanfordcorenlp import StanfordCoreNLP

class CustomizedSequences(Dataset):
    def __init__(self, all_pairs_file_path, tokenizer, train_mode):

        self.train_mode = train_mode
        self.all_pairs = []
        with open(all_pairs_file_path, "rb") as f:
          for record in ijson.items(f, "item"):
            self.all_pairs.append(record)
        f.close()

        self.tokenizer = tokenizer


    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, idx):

        img_size = 64

        positive_sample = np.random.choice([0,1],p=[0.5,0.5])

        count = 0
        out_item = {}
        img_list = torch.zeros((64,3,img_size,img_size))
        sentences = self.all_pairs[idx][0].copy()
        ners = self.all_pairs[idx][1].copy()
        img_temp_list = torch.zeros((2, 3, img_size, img_size))
        img_ner_index = []
        ner_onehot_embedding = torch.zeros((64,5))
        ner_temp_list = torch.zeros((2,5))
        
        if self.train_mode == 'pretrain':
          if positive_sample == 0:
            sentences.reverse()
            ners.reverse()
            out_item['label'] = 0
          else:
            out_item['label'] = 1
        
        if self.train_mode == 'finetune':
          if positive_sample == 0:
            random_index = np.random.randint(len(self.all_pairs))
            random_sentences = self.all_pairs[random_index][0].copy()
            random_ners = self.all_pairs[random_index][1].copy()
            sentences[0] = random_sentences[0]
            ners[0] = random_ners[0]
            out_item['label'] = 0
          else:
            out_item['label'] = 1
            

        for (idx2, ner) in enumerate(ners):
            if ner == ['CLIP_IMAGE']:
                img_ner_index.append(idx2)
                img = cv2.imread(sentences[idx2][0])
                img = cv2.resize(img, (img_size, img_size))
                trans = transforms.ToTensor()
                img = trans(img)
                sentences[idx2] = [str(count)]
                count = count + 1
                img_temp_list[idx2, :] = img
                ner_temp_list[idx2, :] = ner_temp_list[idx2, :] + torch.tensor([0,0,0,0,1])
            else:
                ner_compos = ner[0].split(';')
                for ner_compo in ner_compos:
                    if 'TIME' in ner_compo or 'DATE' in ner_compo or 'DURATION' in ner_compo:
                        ner_temp_list[idx2, :] = ner_temp_list[idx2, :] + torch.tensor([0,0,0,1,0])
                    if 'PERSON' in ner_compo:
                        ner_temp_list[idx2, :] = ner_temp_list[idx2, :] + torch.tensor([0,0,1,0,0])
                    if 'LOCATION' in ner_compo or 'CITY' in ner_compo or 'STATE_OR_PROVINCE' in ner_compo or 'COUNTRY' in ner_compo:
                        ner_temp_list[idx2, :] = ner_temp_list[idx2, :] + torch.tensor([0,1,0,0,0])
                    if 'EMAIL' in ner_compo or 'URL' in ner_compo:
                        ner_temp_list[idx2, :] = ner_temp_list[idx2, :] + torch.tensor([1,0,0,0,0])
        output = self.tokenizer(sentences[0][0], sentences[1][0], add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                truncation=True,
                                max_length=64,  # Pad & truncate all sentences.
                                padding='max_length',
                                return_attention_mask=True,
                                return_tensors='pt')
        out_item['input_ids'] = output['input_ids']
        SEP_index = (out_item['input_ids'] == 102).nonzero()[:, 1]

        ner_onehot_embedding[0:SEP_index[0]] = ner_temp_list[0]
        ner_onehot_embedding[SEP_index[0] + 1:SEP_index[1]] = ner_temp_list[1]
        for i in range(len(img_ner_index)):
            img_list[SEP_index[img_ner_index[i]] - 1, :] = img_temp_list[img_ner_index[i]]
        out_item['attention_masks'] = output['attention_mask']
        out_item['sentences'] = sentences
        out_item['ners'] = ners
        out_item['images'] = img_list
        out_item['ners_embeddings'] = ner_onehot_embedding
        return out_item


if __name__ == "__main__":
    all_pairs_file_path = '../all_pairs_no_duplication.json'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    sequences_dataset = CustomizedSequences(all_pairs_file_path, tokenizer, 'finetune')
    print(len(sequences_dataset))