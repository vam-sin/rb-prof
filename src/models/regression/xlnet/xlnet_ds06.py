# libraries
import numpy as np
import torch
from torch.utils.data import random_split
from transformers import XLNetConfig, XLNetForTokenClassification
from model_utils_ds06 import RBDataset_NoBS, RegressionTrainer  # custom dataset and trainer
from transformers import TrainingArguments
from scipy.stats import pearsonr
import wandb
import random
from os import listdir
from os.path import isfile, join

# hyperparameters
lr = 2e-5
architecture = 'XL-Net'
dataset_name = 'DS06_Liver06'
feature_list = ['nt', 'cbert', 'conds']
feature_string = '_'.join(feature_list)
loss_func_name = 'PCCLoss'
epochs = 200
bs = 1
saved_files_name = 'XLNet-3_8_256_808-DS06_Liver06-NT_CBERT-BS16-PCCLoss-dr01'

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="rb-prof",

    # name the run
    name = saved_files_name,
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": architecture,
    "features": feature_string,
    "dataset": dataset_name,
    "epochs": epochs,
    "batch_size": bs,
    "loss": loss_func_name
    }
)

# reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# no af2 data
no_af2_transcripts = ['ENSMUST00000110336.3', 'ENSMUST00000049149.14', 'ENSMUST00000114036.8', 'ENSMUST00000092956.2', 'ENSMUST00000028829.12', 'ENSMUST00000021471.12']

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
elif dataset_name == 'DS06_Liver06':
    train_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/train'
    val_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/val'
    test_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/test'
    liver_files_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper/final/liver_06'
elif dataset_name == 'DS04':
    train_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/train'
    val_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/val'
    test_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/rb_prof_Naef/processed_full_proper_thresh04/final/test'

print("Starting")

conditions_list = []

# train data
mypath = train_path
onlyfiles_full = [f for f in listdir(mypath) if isfile(join(mypath, f))]

onlyfiles = []
for x in onlyfiles_full:
    if x.split('_')[0] not in no_af2_transcripts:
        onlyfiles.append(x)

train_files = [mypath + '/' + f for f in onlyfiles]

if dataset_name == 'DS06_Liver06':
    # liver data
    onlyfiles_liver = [f for f in listdir(liver_files_path) if isfile(join(liver_files_path, f))]
    onlyfiles_full_liver = [liver_files_path + '/' + f for f in onlyfiles_liver]

    full_training_files = [] 
    for x in train_files:
        full_training_files.append(x)
    for x in onlyfiles_full_liver:
        full_training_files.append(x)

    # shuffle train files
    random.shuffle(full_training_files)

    train_data = RBDataset_NoBS(full_training_files, dataset_name, feature_list)

if dataset_name == 'DS06' or dataset_name == 'DS04':
    train_data = RBDataset_NoBS(train_files, dataset_name, feature_list)

print("Loaded Train")

# val data
mypath = val_path
onlyfiles_full = [f for f in listdir(mypath) if isfile(join(mypath, f))]

onlyfiles = []
for x in onlyfiles_full:
    if x.split('_')[0] not in no_af2_transcripts:
        onlyfiles.append(x)

val_files = [mypath + '/' + f for f in onlyfiles]

val_data = RBDataset_NoBS(val_files, dataset_name, feature_list)

print("Loaded Val")

# test data
mypath = test_path
onlyfiles_full = [f for f in listdir(mypath) if isfile(join(mypath, f))]

onlyfiles = []
for x in onlyfiles_full:
    if x.split('_')[0] not in no_af2_transcripts:
        onlyfiles.append(x)

test_files = [mypath + '/' + f for f in onlyfiles]

test_data = RBDataset_NoBS(test_files, dataset_name, feature_list)

print("Loaded Test")

print(f'Train Set: {len(train_data):5d} || Validation Set: {len(val_data):5d} || Test Set: {len(test_data): 5d}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

# load xlnet to train from scratch
config = XLNetConfig(vocab_size=126, pad_token_id=125, d_model = 808, n_layer = 3, n_head = 8, d_inner = 256, num_labels = 1, dropout = 0.1) # 5^3 + 1 for padding
model = XLNetForTokenClassification(config)
# model.word_embedding = torch.nn.Linear(input_size, 128) # 5^3 + 1 for padding
model.classifier = torch.nn.Linear(808, 1, bias=True) # d_model = 128

# data collator
def data_collator(features):
    batch = {}
    batch["inputs_embeds"] = torch.stack([f[0] for f in features])
    batch["labels"] = torch.stack([f[1] for f in features])
    # print(batch["labels"]) # these are alright, the proper labels
    # provide sequence mask
    batch["attention_mask"] = torch.stack([f[2] for f in features])

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

# train xlnet
training_args = TrainingArguments(
    output_dir=saved_files_name,
    learning_rate=2e-5,
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=1,
    num_train_epochs=epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = RegressionTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# save best model
trainer.save_model(saved_files_name + '/best_model')

# evaluate model
trainer.evaluate()
