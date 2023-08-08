# libraries
import numpy as np
import torch
from torch.utils.data import random_split, ConcatDataset
from transformers import XLNetConfig, XLNetForTokenClassification
from model_utils_pretrain import RiboDatasetLiver, RegressionTrainer, RBDataset_NoBS  # custom dataset and trainer
from transformers import TrainingArguments
from scipy.stats import pearsonr
import random
from os import listdir
from os.path import isfile, join

# reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

print("Starting")

# load data
data_path = '/nfs_home/craigher/scratch/translation_proj/data/liver'

# load data
liver_dataset = RiboDatasetLiver(data_path)

print("Loaded Liver Dataset")

# # split into train, test, and val
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size 

# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# no af2 data
no_af2_transcripts = ['ENSMUST00000110336.3', 'ENSMUST00000049149.14', 'ENSMUST00000114036.8', 'ENSMUST00000092956.2', 'ENSMUST00000028829.12', 'ENSMUST00000021471.12']
feature_list = ['cembeds']
dataset_name = 'DS06'

# input size
input_size = 0
if 'nt' in feature_list:
    input_size += 15
if 'cbert' in feature_list:
    input_size += 768
if 'conds' in feature_list:
    input_size += 22
if 't5' in feature_list:
    input_size += 1024
if 'lem' in feature_list:
    input_size += 15
if 'AF2-SS' in feature_list:
    input_size += 5
if 'mlm_cdna_nt_idai' in feature_list:
    input_size += 7680
if 'mlm_cdna_nt_pbert' in feature_list:
    input_size += 1536
if 'cembeds' in feature_list:
    input_size += 256
if 'geom' in feature_list:
    input_size += 8
if 'codon_epa_encodings' in feature_list:
    input_size += 255

if dataset_name == 'DS06':
    train_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/train'
    val_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/val'
    test_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/test'
elif dataset_name == 'DS04':
    train_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/train'
    val_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/val'
    test_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/test'

conditions_list = []

# train data
mypath = train_path
onlyfiles_full = [f for f in listdir(mypath) if isfile(join(mypath, f))]

onlyfiles = []
for x in onlyfiles_full:
    if x.split('_')[0] not in no_af2_transcripts and '_CTRL_' in x:
        onlyfiles.append(x)

train_files = [mypath + '/' + f for f in onlyfiles]

train_data = RBDataset_NoBS(train_files, dataset_name, feature_list)

print("Loaded Train")

# val data
mypath = val_path
onlyfiles_full = [f for f in listdir(mypath) if isfile(join(mypath, f))]

onlyfiles = []
for x in onlyfiles_full:
    if x.split('_')[0] not in no_af2_transcripts and '_CTRL_' in x:
        onlyfiles.append(x)

val_files = [mypath + '/' + f for f in onlyfiles]

val_data = RBDataset_NoBS(val_files, dataset_name, feature_list)

print("Loaded Val")

# test data
mypath = test_path
onlyfiles_full = [f for f in listdir(mypath) if isfile(join(mypath, f))]

onlyfiles = []
for x in onlyfiles_full:
    if x.split('_')[0] not in no_af2_transcripts and '_CTRL_' in x:
        onlyfiles.append(x)

test_files = [mypath + '/' + f for f in onlyfiles]

test_dataset = RBDataset_NoBS(test_files, dataset_name, feature_list)

print("Loaded Test")

print("Loaded DS06 Dataset")

# merge liver, train, and val to make the training dataset
train_dataset = ConcatDataset([liver_dataset, train_data, val_data])

# load xlnet to train from scratch
config = XLNetConfig(vocab_size=126, pad_token_id=125, d_model = 512, n_layer = 3, n_head = 8, d_inner = 512, num_labels = 1) # 5^3 + 1 for padding
model = XLNetForTokenClassification(config)
model.classifier = torch.nn.Linear(512, 1, bias=True) # d_model = 128

# data collator
def data_collator(features):
    batch = {}
    batch["input_ids"] = torch.stack([f[0] for f in features])
    batch["labels"] = torch.stack([f[1] for f in features])
    # print(batch["labels"]) # these are alright, the proper labels
    return batch

# compute metrics
def compute_metrics(pred):
    labels = pred.label_ids # the labels are not ok, becoming -100 for some reason [that was the padding issue, the evaluate func was padding on it's own with -100]
    preds = pred.predictions
    # remove dim 2 for preds
    preds = np.squeeze(preds, axis=2)
    corr_lis = []
    for i in range(len(labels)):
        mask = labels[i] != -100
        # len mask where mask is true
        num_true = len(mask[mask == True])

        pred_sample = preds[i][:num_true]
        label_sample = labels[i][:num_true]
        corr, _ = pearsonr(pred_sample, label_sample)
        corr_lis.append(corr)
    # return the pearson correlation coefficient
    return {"r": np.mean(corr_lis)}

output_loc = "XLNet-3_512_8-DS06TrainValLiver06_DS06Test-Pretrain-BS1"

# train xlnet
training_args = TrainingArguments(
    output_dir=output_loc,
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=100,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = RegressionTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# save best model
trainer.save_model(output_loc + "/best_model")

# evaluate model
trainer.evaluate()
