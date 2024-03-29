'''
double head models
'''

# libraries
import numpy as np
import pandas as pd 
import torch
from transformers import XLNetConfig, XLNetForTokenClassification
from utils_doubleHead_Deprivation import RegressionTrainer, RiboDatasetGWS, GWSDatasetFromPandas  # custom dataset and trainer
from transformers import TrainingArguments
import random
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.functional import pearson_corrcoef
from torchmetrics import Metric
from prediction_utils import analyse_dh_outputs

# reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# dataset paths 
ribo_data_gws = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/AA_depr_full/liver_na_84.csv'
depr_folder = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/AA_depr_full/'
model_name = 'XLNetDepr-3_256_8-TrueALLWithLIVER_Standard-PEL-BS1-TrueGWS_PCC_IndTokens384_doubleheads_3Loss_NonZero20_PercNansThresh0.05_fixedLEU_ILE'

# GWS dataset
ds = 'ALL' # this uses both liver and deprivation datasets all the conditions
train_dataset, test_dataset = RiboDatasetGWS(ribo_data_gws, depr_folder, ds, threshold=0.3, longZerosThresh=20, percNansThresh=0.05)

# convert to torch dataset
train_dataset = GWSDatasetFromPandas(train_dataset)
test_dataset = GWSDatasetFromPandas(test_dataset)

print("samples in train dataset: ", len(train_dataset))
print("samples in test dataset: ", len(test_dataset))

# load xlnet to train from scratch
# GWS
config = XLNetConfig(vocab_size=385, pad_token_id=384, d_model = 256, n_layer = 3, n_head = 8, d_inner = 256, num_labels = 1, dropout=0.3) # 5^3 + 1 for padding
model = XLNetForTokenClassification(config)
# modify the input layer to take 384 to 256
model.classifier = torch.nn.Linear(256, 2, bias=True)

class CorrCoef(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("corrcoefs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
    def update(self, preds, target, mask):
        preds = torch.sum(preds, dim=2)
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
    x, y, ctrl_y, gene, transcript = zip(*batch)

    # sequence lenghts 
    lengths = torch.tensor([len(x) for x in x])
    
    x = pad_sequence(x, batch_first=True, padding_value=384) 
    y = pad_sequence(y, batch_first=True, padding_value=-1)
    ctrl_y = pad_sequence(ctrl_y, batch_first=True, padding_value=-1)

    out_batch = {}

    out_batch["input_ids"] = x
    out_batch["labels"] = y
    out_batch["lengths"] = lengths
    out_batch["labels_ctrl"] = ctrl_y

    return out_batch

# compute metrics
def compute_metrics(pred):
    labels = pred.label_ids 
    preds = pred.predictions
    inputs = pred.inputs
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

# compute metrics
def compute_metrics_saved(pred):
    '''
    additional function to just save everything to do analysis later
    '''
    labels = pred.label_ids 
    preds = pred.predictions
    inputs = pred.inputs
    mask = labels != -100.0
    labels = torch.tensor(labels)
    preds = torch.tensor(preds)
    preds = torch.squeeze(preds, dim=2)
    mask = torch.tensor(mask)
    # mask = torch.arange(preds.shape[1])[None, :].to(lengths) < lengths[:, None]
    mask = torch.logical_and(mask, torch.logical_not(torch.isnan(labels)))
    corr_coef = CorrCoef()
    corr_coef.update(preds, labels, mask)

    # save predictions
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    np.save("preds/" + model_name + "/preds.npy", preds)
    np.save("preds/" + model_name + "/labels.npy", labels)
    np.save("preds/" + model_name + "/inputs.npy", inputs)

    return {"r": corr_coef.compute()}

output_loc = "saved_models/" + model_name

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
    dataloader_num_workers=4,
    include_inputs_for_metrics=True
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

# # evaluate model
trainer.evaluate()

# # # load model best weights
# model.load_state_dict(torch.load(output_loc + "/checkpoint-1806186/pytorch_model.bin"))

# trainer.evaluate()

# # # analyse preds
# preds = np.load("preds/" + model_name + "/preds.npy")
# labels = np.load("preds/" + model_name + "/labels.npy")
# inputs = np.load("preds/" + model_name + "/inputs.npy")

# analyse_dh_outputs(preds, labels, inputs, "preds/" + model_name + "/analysis_dh")

# '''
# ALL means ALL including liver and deprivation ds
# '''

# '''
# from the analysis: XLNetDepr-3_256_8-ALL_Standard-PEL-BS1-TrueGWS_PCC_IndTokens384_doubleheads

# 1. ctrl predictions: the depr diff is not zero BUT the final and ctrl predictions have pretty much same pcc
# 2. the ctrl predictions always have to be positive so maybe we can add a activation on that specifically
# 3. additional loss on just the ctrl predictions
# 4. looks like both of them are making the same predictions, maybe we have to force it to learn the ctrl and the depr diff separately
# '''

'''
fixed mse issues
new model with double heads and 384 tokens (ALL of the data)
'''