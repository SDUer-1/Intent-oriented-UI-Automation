import torch.nn as nn
import numpy as np
import torchvision
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DimensionReduction(nn.Module):
  def __init__(self, input_dimension, target_dimension):
    super(DimensionReduction, self).__init__()
    self.dimension_reduction_layer = nn.Sequential(
        nn.Linear(input_dimension, 512),
        nn.ReLU(),
        nn.Linear(512, target_dimension),
        nn.ReLU()
    )
  def forward(self, x):
    return self.dimension_reduction_layer(x)

class UITextEncoder(nn.Module):
    def __init__(self, bert_base):
        super(UITextEncoder, self).__init__()
        self.bert_base = bert_base
        # self.image_encoder = ImageEncoder()
        self.word_embedding_layer = self.bert_base.embeddings
        self.dimension_reduction = DimensionReduction(768, 256)
        # self.ln = torch.nn.LayerNorm(normalized_shape=[64, 768])

    def forward(self, input_ids, input_masks, input_imgs):
        word_embeddings = self.word_embedding_layer(input_ids, input_masks)
        batch_size = input_imgs.size()[0]
        img_0_position_check_tensor = 1014 * torch.ones((batch_size, 64)).to(device)
        img_1_position_check_tensor = 1015 * torch.ones((batch_size, 64)).to(device)
        non_word_position = (input_ids == img_0_position_check_tensor) | (input_ids == img_1_position_check_tensor)
        word_embeddings[non_word_position] = torch.zeros((768)).to(device)

        # sentence_lenth = input_imgs.size()[1]
        # input_imgs = input_imgs.reshape(
        #    (batch_size * sentence_lenth, input_imgs.size()[2], input_imgs.size()[3], input_imgs.size()[4]))
        # img_embeddings = self.image_encoder(input_imgs)
        # img_embeddings = img_embeddings.reshape((batch_size, sentence_lenth, img_embeddings.size()[1]))
        # word_embeddings = self.ln(word_embeddings)
        # img_embeddings = self.ln(img_embeddings)
        # input_embeddings = word_embeddings + img_embeddings
        output = self.bert_base(inputs_embeds=word_embeddings, attention_mask=input_masks)[1]
        output = self.dimension_reduction(output)
        return output



class NerEncoder(nn.Module):
    def __init__(self):
        super(NerEncoder, self).__init__()
        self.ner_embedding_layer = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU()
        )

    def forward(self, ner):
        return self.ner_embedding_layer(ner)




class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.resnet_encoder = torchvision.models.resnet18()
        fc_in_features = self.resnet_encoder.fc.in_features
        self.resnet_encoder.fc = nn.Linear(fc_in_features, 256)

    def forward(self, img):

        return self.resnet_encoder(img)

class CLIP(nn.Module):
    def __init__(self, former_bert_base, latter_bert_base, temperature):
        super(CLIP, self).__init__()
        self.former_bert_base = former_bert_base
        self.latter_bert_base = latter_bert_base
        self.former_encoder = UITextEncoder(self.former_bert_base)
        self.latter_encoder = UITextEncoder(self.latter_bert_base)
        self.image_encoder = ImageEncoder()
        self.ner_encoder = NerEncoder()
        self.ln = torch.nn.LayerNorm(normalized_shape=256)
        self.t = temperature

    def forward(self, former_input_ids, latter_input_ids, former_attention_masks, latter_attention_masks, former_images, latter_images, former_ners, latter_ners):

        Former_feature = self.former_encoder(former_input_ids, former_attention_masks, former_images)
        Latter_feature = self.latter_encoder(latter_input_ids, latter_attention_masks, latter_images)
        former_ners_embeddings = self.ner_encoder(former_ners)
        latter_ners_embeddings = self.ner_encoder(latter_ners)

        # for images
        input_size = 64
        batch_size = former_images.size()[0]
        img_0_position_check_tensor = 1014 * torch.ones((batch_size, 64)).to(device)
        img_1_position_check_tensor = 1015 * torch.ones((batch_size, 64)).to(device)
        former_non_word_position = (former_input_ids == img_0_position_check_tensor) | (former_input_ids == img_1_position_check_tensor)
        latter_non_word_position = (latter_input_ids == img_0_position_check_tensor) | (latter_input_ids == img_1_position_check_tensor)

        former_img_list = torch.zeros((batch_size, 3, input_size, input_size)).to(device)
        former_img_list[former_non_word_position.nonzero()[:,0]] = former_images[former_non_word_position]
        latter_img_list = torch.zeros((batch_size, 3, input_size, input_size)).to(device)
        latter_img_list[latter_non_word_position.nonzero()[:,0]] = latter_images[latter_non_word_position]

        former_img_embeddings = self.image_encoder(former_img_list)
        latter_img_embeddings = self.image_encoder(latter_img_list)

        # Layer Normalization
        Former_feature = self.ln(Former_feature)
        Latter_feature = self.ln(Latter_feature)
        former_ners_embeddings = self.ln(former_ners_embeddings)
        latter_ners_embeddings = self.ln(latter_ners_embeddings)
        former_img_embeddings = self.ln(former_img_embeddings)
        latter_img_embeddings = self.ln(latter_img_embeddings)


        # summing
        # former_embedding = Former_feature + former_ners_embeddings + former_img_embeddings
        # latter_embedding = Latter_feature + latter_ners_embeddings + latter_img_embeddings
        
        # concat
        former_embedding = torch.cat((Former_feature, former_img_embeddings),1)
        former_embedding = torch.cat((former_embedding, former_ners_embeddings),1)
        latter_embedding = torch.cat((Latter_feature, latter_img_embeddings),1)
        latter_embedding = torch.cat((latter_embedding, latter_ners_embeddings),1)

        logits = torch.matmul(former_embedding, latter_embedding.T) * np.exp(self.t)
        
        # 3 matrices
        # logits_T = torch.matmul(Former_feature, Latter_feature.T) * np.exp(self.t)
        # logits_I = torch.matmul(former_img_embeddings, latter_img_embeddings.T) * np.exp(self.t)
        # logits_N = torch.matmul(former_ners_embeddings, latter_ners_embeddings.T) * np.exp(self.t)

        return logits

