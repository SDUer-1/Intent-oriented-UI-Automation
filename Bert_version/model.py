import torch
import torchvision
import torch.nn as nn
from transformers import BertForNextSentencePrediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BertNSPM(nn.Module):
    def __init__(self, bert_base):
        super(BertNSPM, self).__init__()
        self.bert_base = bert_base
        self.resnet_encoder = torchvision.models.resnet18()
        fc_in_features = self.resnet_encoder.fc.in_features
        self.resnet_encoder.fc = nn.Linear(fc_in_features, 768)
        self.word_embedding_layer = self.bert_base.bert.embeddings
        self.ner_embedding_layer = nn.Sequential(
            nn.Linear(5,64),
            nn.ReLU(),
            nn.Linear(64,768),
            nn.ReLU()
        )
        self.ln = torch.nn.LayerNorm(normalized_shape=[64, 768])

    def forward(self, input_ids, input_masks, input_imgs, input_ners, input_ners_embeddings):
        word_embeddings = self.word_embedding_layer(input_ids, input_masks)
        batch_size = input_imgs.size()[0]
        img_0_position_check_tensor = 1014 * torch.ones((batch_size, 64)).to(device)
        img_1_position_check_tensor = 1015 * torch.ones((batch_size, 64)).to(device)
        non_word_position = (input_ids == img_0_position_check_tensor) | (input_ids == img_1_position_check_tensor)
        word_embeddings[non_word_position] = torch.zeros((768)).to(device)
        
        
        sentence_lenth = input_imgs.size()[1]
        input_imgs = input_imgs.reshape((batch_size * sentence_lenth, input_imgs.size()[2], input_imgs.size()[3], input_imgs.size()[4]))
        img_embeddings = self.resnet_encoder(input_imgs)
        img_embeddings = img_embeddings.reshape((batch_size, sentence_lenth, img_embeddings.size()[1]))
        ner_embedding = self.ner_embedding_layer(input_ners_embeddings)
        
        word_embeddings = self.ln(word_embeddings)
        img_embeddings = self.ln(img_embeddings)
        ner_embedding = self.ln(ner_embedding)
        input_embeddings = word_embeddings + img_embeddings + ner_embedding
        return self.bert_base(inputs_embeds=input_embeddings, attention_mask=input_masks)