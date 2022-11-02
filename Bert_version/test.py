from transformers import  BertTokenizer, BertForNextSentencePrediction
import torch

# train_input = [[['OK'], ['Allow'], ['Illinois;Champaign;Illinois;Urbana;Illinois;Wilbur Heights;Illinois;Augerville;Illinois;Kenwood;Illinois;Mira;Illinois;Savoy;Illinois;Staley'], ['Tomorrow;3 JUN .;Chance of rain;Rain;Maximum gusts;Wind - average;Search;Share;Menu;Swipe up and down'], ['scroll'], ['scroll'], ['Snow expected in the next few hours'], ['Detailed information;08:00;South;Swipe sideways;Hourly graphic'], ['D:\\MoTIF/raw_data\\aplicacion.tiempo\\09ca3d68-ae8d-4421-9fe0-3ff717442bd1\\clips\\b61fb92e-baae-44de-b00b-93a384026a90_247_1612645566882.jpg'], ['D:\\MoTIF/raw_data\\aplicacion.tiempo\\09ca3d68-ae8d-4421-9fe0-3ff717442bd1\\clips\\b61fb92e-baae-44de-b00b-93a384026a90_260_1612645568191.jpg'], ['scroll'], ['scroll'], ['scroll'], ['scroll'], ['scroll'], ['scroll'], ['scroll'], ['scroll'], ['scroll'], ['scroll'], ['scroll'], ['scroll'], ['scroll'], ['scroll'], ['scroll'], ['Navigate up'], ['Weather Radar'], ['D:\\MoTIF/raw_data\\aplicacion.tiempo\\09ca3d68-ae8d-4421-9fe0-3ff717442bd1\\clips\\b61fb92e-baae-44de-b00b-93a384026a90_597_1612645707796.jpg'], ['Forecast Maps'], ['Navigate up'], ['My locations'], ['D:\\MoTIF/raw_data\\aplicacion.tiempo\\09ca3d68-ae8d-4421-9fe0-3ff717442bd1\\clips\\b61fb92e-baae-44de-b00b-93a384026a90_715_1612645722249.jpg'], ['Discover another way to check the weather;TURN YOUR PHONE;OK']], [['O'], ['O'], ['STATE_OR_PROVINCE;CITY;STATE_OR_PROVINCE;CITY;STATE_OR_PROVINCE;PERSON PERSON;STATE_OR_PROVINCE;PERSON;STATE_OR_PROVINCE;PERSON;STATE_OR_PROVINCE;PERSON;STATE_OR_PROVINCE;PERSON;STATE_OR_PROVINCE;PERSON'], ['DATE;DATE DATE O;O O O;O;O O;O O O;O;O;O;O O O O'], ['OPERATION'], ['OPERATION'], ['O O O DURATION DURATION DURATION DURATION'], ['O O;TIME;O;O O;SET O'], ['CLIP_IMAGE'], ['CLIP_IMAGE'], ['OPERATION'], ['OPERATION'], ['OPERATION'], ['OPERATION'], ['OPERATION'], ['OPERATION'], ['OPERATION'], ['OPERATION'], ['OPERATION'], ['OPERATION'], ['OPERATION'], ['OPERATION'], ['OPERATION'], ['OPERATION'], ['OPERATION'], ['O O'], ['O O'], ['CLIP_IMAGE'], ['O O'], ['O O'], ['O O'], ['CLIP_IMAGE'], ['O O O O O O O;O O O;O']]]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
text_A = 'OK'
text_B = 'Allow'
inputs = tokenizer(text_A, text_B, truncation=True,
            max_length=64,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,return_tensors='pt')

input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
print(input_ids.size())
print(attention_mask.size())
labels = torch.LongTensor([1])
outputs = model(input_ids, attention_mask)
print(outputs)
