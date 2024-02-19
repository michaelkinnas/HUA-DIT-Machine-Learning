import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

torch.manual_seed(42)

# See https://huggingface.co/docs/transformers/main_classes/tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')

device = torch.device("cuda")
# device = torch.device("cpu")

configuration = GPT2Config.from_pretrained('./model_save/', output_hidden_states=False)
model = GPT2LMHeadModel.from_pretrained("./model_save/", config=configuration)
model.resize_token_embeddings(len(tokenizer))

# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())
print('The GPT-2 model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')
for p in params[0:2]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')
for p in params[2:14]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')
for p in params[-2:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))