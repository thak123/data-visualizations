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

PTM =  "gilf/english-yelp-sentiment"
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

# all_logits = np.empty([0,2])

extracted_features = []

for batch in dataloader_val:

    batch = tuple(b.to(device) for b in batch)
    inputs = {'input_ids': batch[0],
              'attention_mask': batch[1],
              }
    with torch.no_grad():
        outputs = model(**inputs)

    extracted_features.extend(
        outputs.pooler_output.detach().cpu().numpy().tolist())

print(len(extracted_features))

mytsne_tokens = TSNE(n_components=2, early_exaggeration=12,
                     verbose=2, metric='cosine', init='pca', n_iter=2500)
embs_tsne = mytsne_tokens.fit_transform(extracted_features)
X = pd.DataFrame(np.concatenate([embs_tsne], axis=1),
                 columns=["x1", "y1"])
X = X.astype({"x1": float, "y1": float})

# Plot for layer -1
plt.figure(figsize=(20, 15))
p1 = sns.scatterplot(x=X["x1"], y=X["y1"], palette="coolwarm")
# p1.set_title("development-"+str(row+1)+", layer -1")
x_texts = []
for value in zip(dataset['label']):
    if 1 == value[0]:
        x_texts.append("@P")
    elif 0 == value[0]:
        x_texts.append("@N")
    else:
        x_texts.append("@U")


X["texts"] = x_texts

for line in X.index:
    text = X.loc[line, "texts"]+"-"+str(line)
    if "@U" in text:
        p1.text(X.loc[line, "x1"]+0.2, X.loc[line, "y1"], text[2:], horizontalalignment='left',
                size='medium', color='blue', weight='semibold')
    elif "@P" in text:
        p1.text(X.loc[line, "x1"]+0.2, X.loc[line, "y1"], text[2:], horizontalalignment='left',
                size='medium', color='green', weight='semibold')
    elif "@N" in text:
        p1.text(X.loc[line, "x1"]+0.2, X.loc[line, "y1"], text[2:], horizontalalignment='left',
                size='medium', color='red', weight='semibold')
    else:
        p1.text(X.loc[line, "x1"]+0.2, X.loc[line, "y1"], text, horizontalalignment='left',
                size='medium', color='black', weight='semibold')
plt.show()
plt.savefig('./figure.svg', format="svg")
plt.savefig('./figure.png', format="png")
