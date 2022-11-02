import json
import cv2
from PIL import Image
import torch
import numpy as np
import copy
from torch.utils.data import Dataset
from transformers import BertTokenizer
import torchvision.transforms as transforms
from stanfordcorenlp import StanfordCoreNLP

class CustomizedSequences(Dataset):
    def __init__(self, all_pairs_file_path, tokenizer):
        with open(all_pairs_file_path, 'r') as p_s:
            self.all_pairs = json.load(p_s)

        self.tokenizer = tokenizer


    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, idx):

        img_size = 64

        count = 0
        former_UI = {}
        latter_UI = {}
        former_img_list = torch.zeros((64, 3, img_size, img_size))
        latter_img_list = torch.zeros((64, 3, img_size, img_size))
        imgs_list = [former_img_list, latter_img_list]
        sentences = self.all_pairs[idx][0].copy()
        ners = self.all_pairs[idx][1].copy()

        # former_UI, latter_UI

        former_ner_onehot_embedding = torch.zeros((5))
        latter_ner_onehot_embedding = torch.zeros((5))
        ner_embeddings = [former_ner_onehot_embedding, latter_ner_onehot_embedding]

        for (idx2, ner) in enumerate(ners):
            if ner == ['CLIP_IMAGE']:
                img = Image.open(sentences[idx2][0])
                img = np.array(img.resize((img_size,img_size)))
                # img = cv2.resize(img, (img_size, img_size))
                trans = transforms.ToTensor()
                img = trans(img)
                sentences[idx2] = [str(count)]
                count = count + 1
                imgs_list[idx2][1, :] = img
                ner_embeddings[idx2] = ner_embeddings[idx2] + torch.tensor([0,0,0,0,1])
            else:
                ner_compos = ner[0].split(';')
                for ner_compo in ner_compos:
                    if 'TIME' in ner_compo or 'DATE' in ner_compo or 'DURATION' in ner_compo:
                        ner_embeddings[idx2] = ner_embeddings[idx2] + torch.tensor([0,0,0,1,0])
                    if 'PERSON' in ner_compo:
                        ner_embeddings[idx2] = ner_embeddings[idx2] + torch.tensor([0,0,1,0,0])
                    if 'LOCATION' in ner_compo or 'CITY' in ner_compo or 'STATE_OR_PROVINCE' in ner_compo or 'COUNTRY' in ner_compo:
                        ner_embeddings[idx2] = ner_embeddings[idx2] + torch.tensor([0,1,0,0,0])
                    if 'EMAIL' in ner_compo or 'URL' in ner_compo:
                        ner_embeddings[idx2] = ner_embeddings[idx2] + torch.tensor([1,0,0,0,0])
        former_output = self.tokenizer(sentences[0][0], add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                truncation=True,
                                max_length=64,  # Pad & truncate all sentences.
                                padding='max_length',
                                return_attention_mask=True,
                                return_tensors='pt')
        latter_output = self.tokenizer(sentences[1][0], add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                          truncation=True,
                                          max_length=64,  # Pad & truncate all sentences.
                                          padding='max_length',
                                          return_attention_mask=True,
                                          return_tensors='pt')
        former_UI['input_ids'] = former_output['input_ids']
        former_UI['attention_masks'] = former_output['attention_mask']
        former_UI['sentences'] = sentences[0]
        former_UI['ners'] = ners[0]
        former_UI['images'] = former_img_list
        former_UI['ners_embeddings'] = former_ner_onehot_embedding
        latter_UI['input_ids'] = latter_output['input_ids']
        latter_UI['attention_masks'] = latter_output['attention_mask']
        latter_UI['sentences'] = sentences[1]
        latter_UI['ners'] = ners[1]
        latter_UI['images'] = latter_img_list
        latter_UI['ners_embeddings'] = latter_ner_onehot_embedding
        return former_UI, latter_UI


if __name__ == "__main__":
    all_pairs_file_path = 'C:\\Users/15091/Desktop/Intent-UI Automation/part_pairs.json'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    sequences_dataset = CustomizedSequences(all_pairs_file_path, tokenizer)
    out = sequences_dataset[-2]
    print(out)