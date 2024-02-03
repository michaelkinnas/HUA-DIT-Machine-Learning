from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token_id = tokenizer.eos_token_id

model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)


# model.to(device)


while True:
    text = input("\nPrompt: > ")

    encoded_text = tokenizer(text, return_tensors='pt').to(device)

    response = model.generate(**encoded_text, max_new_tokens=200, do_sample=True, top_k=20)

    response_text = tokenizer.decode(response[0], skip_special_tokens=True)

    print(response_text)