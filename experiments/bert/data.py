from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
import torch


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = tokenizer.tokenize(text)
# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
print(tokenized_text)

# Load pre-trained model (weights)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

model.eval()

# Predict hidden states features for each layer
with torch.no_grad():
    output = model(tokens_tensor, segments_tensors)
# We have a hidden states for each of the 12 layers in model bert-base-uncased
assert output.size[-1] == 5


# # Load pre-trained model (weights)
# model = BertForMaskedLM.from_pretrained('bert-base-uncased')
# model.eval()

# # Tokenized input
# text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
# tokenized_text = tokenizer.tokenize(text)
#
# # Mask a token that we will try to predict back with `BertForMaskedLM`
# masked_index = 8
# tokenized_text[masked_index] = '[MASK]'
# assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']
#
# # Convert token to vocabulary indices
# indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
# segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
#
# # Convert inputs to PyTorch tensors
# tokens_tensor = torch.tensor([indexed_tokens])
# segments_tensors = torch.tensor([segments_ids])
#
#
# # Predict all tokens
# with torch.no_grad():
#     predictions = model(tokens_tensor, segments_tensors)
#
# # confirm we were able to predict 'henson'
# predicted_index = torch.argmax(predictions[0, masked_index]).item()
# predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
# assert predicted_token == 'henson'

## Experiment
# text = "[CLS] A new release of BERT (Devlin, 2018) includes a model simultaneously pretrained on 104 languages with impressive performance for zero-shot cross-lingual transfer on a natural language inference task. [SEP]"
# tokenized_text = tokenizer.tokenize(text)
# print(tokenized_text)
# masked_index = 3
# tokenized_text[masked_index] = '[MASK]'
# print(tokenized_text)
# indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# segments_ids = [0]*len(tokenized_text)
# tokens_tensor = torch.tensor([indexed_tokens])
# segments_tensors = torch.tensor([segments_ids])
# # Predict all tokens
# with torch.no_grad():
#     predictions = model(tokens_tensor, segments_tensors)
#
# predicted_index = torch.argmax(predictions[0, masked_index]).item()
# predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
# print(predicted_token)


# from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
#
# # OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
# import logging
# logging.basicConfig(level=logging.INFO)
#
# # Load pre-trained model tokenizer (vocabulary)
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#
# # Encode some inputs
# text_1 = "Who was Jim Henson ?"
# text_2 = "Jim Henson was a puppeteer"
# indexed_tokens_1 = tokenizer.encode(text_1)
# indexed_tokens_2 = tokenizer.encode(text_2)
#
# # Convert inputs to PyTorch tensors
# tokens_tensor_1 = torch.tensor([indexed_tokens_1])
# tokens_tensor_2 = torch.tensor([indexed_tokens_2])
#
# # Load pre-trained model (weights)
# model = GPT2Model.from_pretrained('gpt2')
# model.eval()
#
# # Predict hidden states features for each layer
# with torch.no_grad():
#     hidden_states_1, past = model(tokens_tensor_1)
#     # past can be used to reuse precomputed hidden state in a subsequent predictions
#     # (see beam-search examples in the run_gpt2.py example).
#     hidden_states_2, past = model(tokens_tensor_2, past=past)