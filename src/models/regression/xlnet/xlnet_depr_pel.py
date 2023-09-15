# libraries
import numpy as np
import pandas as pd 
import torch
from transformers import XLNetConfig, XLNetForTokenClassification
from model_utils_pel_depr import RegressionTrainer, RiboDatasetGWS, GWSDatasetFromPandas  # custom dataset and trainer
from transformers import TrainingArguments
import random
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.functional import pearson_corrcoef
from torchmetrics import Metric

# reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

ribo_data_gws = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/AA_depr_full/liver.csv'
depr_folder = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/AA_depr_full/'

# GWS dataset
ds = 'ALL'
train_dataset, test_dataset = RiboDatasetGWS(ribo_data_gws, depr_folder, ds, threshold=0.3)

# convert to torch dataset
train_dataset = GWSDatasetFromPandas(train_dataset)
test_dataset = GWSDatasetFromPandas(test_dataset)

print("samples in train dataset: ", len(train_dataset))
print("samples in test dataset: ", len(test_dataset))

# load xlnet to train from scratch
# GWS
config = XLNetConfig(vocab_size=385, pad_token_id=384, d_model = 256, n_layer = 3, n_head = 8, d_inner = 256, num_labels = 1) # 5^3 + 1 for padding
model = XLNetForTokenClassification(config)
# modify the input layer to take 384 to 256
model.classifier = torch.nn.Linear(256, 1, bias=True)

class CorrCoef(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("corrcoefs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
    def update(self, preds, target, mask):
        assert preds.shape == target.shape
        assert preds.shape == mask.shape
        coeffs = []
        for p, t, m in zip(preds, target, mask):
            mp, mt = torch.masked_select(p, m), torch.masked_select(t, m)
            temp_pearson = pearson_corrcoef(mp, mt)
            coeffs.append(temp_pearson)
        coeffs = torch.stack(coeffs)
        self.corrcoefs += torch.sum(coeffs)
        self.total += len(coeffs)
    def compute(self):
        return self.corrcoefs / self.total

# collate function
def collate_fn(batch):
    # batch is a list of tuples (x, y)
    x, y = zip(*batch)

    # sequence lenghts 
    lengths = torch.tensor([len(x) for x in x])
    
    x = pad_sequence(x, batch_first=True, padding_value=384) 
    y = pad_sequence(y, batch_first=True, padding_value=-1)

    out_batch = {}

    out_batch["input_ids"] = x
    out_batch["labels"] = y
    out_batch["lengths"] = lengths

    return out_batch

# compute metrics
def compute_metrics(pred):
    labels = pred.label_ids 
    preds = pred.predictions
    mask = labels != -100.0
    labels = torch.tensor(labels)
    preds = torch.tensor(preds)
    preds = torch.squeeze(preds, dim=2)
    mask = torch.tensor(mask)
    # mask = torch.arange(preds.shape[1])[None, :].to(lengths) < lengths[:, None]
    mask = torch.logical_and(mask, torch.logical_not(torch.isnan(labels)))
    corr_coef = CorrCoef()
    corr_coef.update(preds, labels, mask)

    return {"r": corr_coef.compute()}

output_loc = "saved_models/XLNetDepr-3_256_8-ALL_Standard-PEL-BS1-TrueGWS_PCC_IndTokens384"

# train xlnet
# save max 5 checkpoints
training_args = TrainingArguments(
    output_dir=output_loc,
    learning_rate=1e-4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=4,
    num_train_epochs=200,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    dataloader_pin_memory=True,
    save_total_limit=5,
    dataloader_num_workers=4
)

trainer = RegressionTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

trainer.train()

# save best model
trainer.save_model(output_loc + "/best_model")

# evaluate model
trainer.evaluate()