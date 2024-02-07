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

model.to(device)
model.eval()

while True:
    text = input("\nPrompt: > ")

    encoded_text = tokenizer(text, return_tensors='pt').to(device)

    output = model.generate(**encoded_text, do_sample=True, top_k=50, max_length=768,top_p=0.95, num_return_sequences=1)

    for i, sample_output in enumerate(output):
        sample_output_dec = tokenizer.decode(sample_output, skip_special_tokens=True)
        print(f"\n{i}: {sample_output_dec}")