import os
import time
import datetime

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

torch.manual_seed(42)

class GPT2Dataset(Dataset):
    def __init__(self, txt_list, tokenizer, gpt2_type="gpt2", max_length=768):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        
        for txt in txt_list:
            encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length") 
        
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]
    

# If the file is not in the same directory, replace the following with the path
filename = 'star_wars.txt'
with open(filename) as file:
    starwars = [line.rstrip() for line in file]

nlines = 8
min_nlines=3
l = len(starwars)
starwars_seq = []
for i in range(l-nlines):
    range_end = min(l-min_nlines,i+nlines)
    interaction = '\n'.join(starwars[i:range_end])
    starwars_seq.append(interaction)

# See https://huggingface.co/docs/transformers/main_classes/tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')

batch_size = 2

dataset = GPT2Dataset(starwars_seq, tokenizer, max_length=768)
dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)

print(f"Number of batches: {len(dataloader)}")

configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
model.resize_token_embeddings(len(tokenizer))

# device = torch.device("cuda")
device = torch.device("cpu")

epochs = 5
learning_rate = 5e-4
warmup_steps = 1e2
epsilon = 1e-8

sample_every = 100

optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon)
total_steps = len(dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

total_t0 = time.time()

training_stats = []

model = model.to(device)


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

# Train the model
model.train()
for epoch_i in range(0, epochs):
    print("")
    print(f'======== Epoch {epoch_i + 1} / {epochs} ========')
    print('Training...')

    t0 = time.time()
    total_train_loss = 0

    for step, batch in enumerate(dataloader):
        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)

        model.zero_grad()

        outputs = model(b_input_ids, labels=b_labels, attention_mask=b_masks, token_type_ids=None)
        
        loss = outputs[0]
        
        batch_loss = loss.item()
        total_train_loss += batch_loss
        
        # Get sample every x batches.
        if step % sample_every == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print(f' Batch {step:>5,} of ' + f'{len(dataloader):>5,}. Loss: {batch_loss:>5,}.' + f'Elapsed: {elapsed}.')
            
            model.eval()

            sample_outputs = model.generate(do_sample=True, top_k=50, max_length=768,top_p=0.95, num_return_sequences=1)
            
            for i, sample_output in enumerate(sample_outputs):
                sample_output_dec = tokenizer.decode(sample_output, skip_special_tokens=True)
                print(f"{i}: {sample_output_dec}")
            
            model.train()

        loss.backward()
        optimizer.step()
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(dataloader)

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print(f" Average training loss: {avg_train_loss:0.2f}")
    print(f" Training epoch took: {training_time}")

print("")
print("Training complete!")
timediff = format_time(time.time()-total_t0)
print(f"Total training took {timediff} (h:mm:ss)")

# Saving best-practices: if you use default names for the model, you can reload
# it using from_pretrained()
output_dir = './model_save/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Saving model to {output_dir}")

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)