#https://www.kaggle.com/tanmay17061/transformers-bert-hidden-embeddings-visualization
import os
import logging
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModel
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import sys
np.set_printoptions(threshold=sys.maxsize)

logging.basicConfig(level=logging.INFO)

dim_reducer = TSNE(n_components=2)


def visualize_layerwise_embeddings(hidden_states, masks, labels, epoch, title, layers_to_visualize):

    os.makedirs("./plots/{}".format(title),exist_ok=True)
    num_layers = len(layers_to_visualize)

    # each subplot of size 6x6, each row will hold 4 plots
    fig = plt.figure(figsize=(24, (num_layers/4)*6))
    ax = [fig.add_subplot(num_layers/4, 4, i+1) for i in range(num_layers)]
    labels = labels.reshape(-1)

    for i, layer_i in enumerate(layers_to_visualize):
        layer_embeds = hidden_states[layer_i]

        layer_averaged_hidden_states = torch.div(
            layer_embeds.sum(dim=1), masks.sum(dim=1, keepdim=True))
        layer_dim_reduced_embeds = dim_reducer.fit_transform(
            layer_averaged_hidden_states.detach().cpu().numpy())

        df = pd.DataFrame.from_dict(
            {'x': layer_dim_reduced_embeds[:, 0], 'y': layer_dim_reduced_embeds[:, 1], 'label': labels})

        sns.scatterplot(data=df, x='x', y='y', hue='label', ax=ax[i])
        fig.suptitle(f"{title}: epoch {epoch}")
        ax[i].set_title(f"layer {layer_i+1}")

    plt.savefig(f'./plots/{title}/{epoch}.png', format='png', pad_inches=0)


PTM = "gilf/english-yelp-sentiment"
config = AutoConfig.from_pretrained(
    PTM, output_hidden_states=True)
model = AutoModel.from_pretrained(
    PTM, config=config)

# Set the model to eval mode.
model.eval()
# This notebook assumes CPU execution. If you want to use GPUs, put the model on cuda and modify subsequent code blocks.
model.to('cuda')
# Load tokenizer.
tokenizer = AutoTokenizer.from_pretrained(PTM)


dataset = load_dataset("yelp_polarity", split=f'train[:500]')
inputs = tokenizer(list(dataset["text"]), padding=True,
                   truncation=True, return_tensors="pt")

input_ids_val = inputs['input_ids']
attention_masks_val = inputs['attention_mask']
batch_size = 16

dataset_val = TensorDataset(input_ids_val, attention_masks_val)
dataloader_val = DataLoader(dataset_val, sampler=SequentialSampler(
    dataset_val), batch_size=batch_size)
device = "cuda"


extracted_features = []

train_hidden_states = None
train_masks = torch.zeros(0, 512)

for batch in dataloader_val:

    batch = tuple(b.to(device) for b in batch)
    inputs = {
        'input_ids': batch[0],
        'attention_mask': batch[1],
    }
    with torch.no_grad():
        outputs = model(**inputs)
    # print((outputs["hidden_states"]))
    # logits = outputs[0]
        print()

        extracted_features.extend(
            outputs.pooler_output.detach().cpu().numpy().tolist())
     
        hidden_states = outputs.hidden_states[1:]
   
        train_masks = torch.cat([train_masks, batch[1].cpu()])
        if type(train_hidden_states) == type(None):
            train_hidden_states = tuple(layer_hidden_states.cpu()
                                        for layer_hidden_states in hidden_states)
        else:
            train_hidden_states = tuple(torch.cat([layer_hidden_state_all, layer_hidden_state_batch.cpu(
            )])for layer_hidden_state_all, layer_hidden_state_batch in zip(train_hidden_states, hidden_states))


visualize_layerwise_embeddings(hidden_states=train_hidden_states,
                               masks=train_masks,
                               labels=np.array(dataset["label"]),
                               epoch=0,
                               title='train_data',
                               layers_to_visualize=[0, 1, 2, 3, 8, 9, 10, 11]
                               )
