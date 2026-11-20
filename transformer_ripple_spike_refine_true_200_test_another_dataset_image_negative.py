import math
import torch
from torch import nn
import torchvision.transforms as transforms
import torch
import torchvision
import torchvision.transforms as transforms
# from SpectroDataset import SpectroDataset
# from RippleDataset import RippleDataset
from RippleTrialDataset_200 import RippleTrialDataset_200
from scipy.io import loadmat
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut

#@title Utils 🛠️
import json, os, math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc, average_precision_score
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt, ticker as mticker


from torch import nn, optim
import cmocean
import cmocean.cm as cmo
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning) 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import h5py

import warnings
warnings.filterwarnings("ignore")
from scipy.io import savemat, loadmat
import os
import random
import glob

# interest_elecs = int(sys.argv[1])


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# batch_size = 32 #@param {type: "integer"}
# epochs = 200 # For channel 38 exclusively 
# lr = 1e-5  
# lr = 5e-6 # For channel 38 exclusively 
lr = 3.6e-6
save_model_every = 0 #@param {type: "integer"}



device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = {
    "patch_size": 4,  # Input image size: 64x64 -> 8x8 patches
    "hidden_size": 12,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "intermediate_size": 4 * 12, # 4 * hidden_size
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 2, # num_classes of CIFAR10
    "num_channels": 1,
    "qkv_bias": True,
    "use_faster_attention": True,
}
# These are not hard constraints, but are used to prevent misconfigurations
assert config["hidden_size"] % config["num_attention_heads"] == 0
assert config['intermediate_size'] == 4 * config['hidden_size']
assert config['image_size'] % config['patch_size'] == 0


class GeluActivation(nn.Module):
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))



class PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        self.num_patches = (self.image_size//self.patch_size) **2
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        x = self.projection(x)
        # print("X: ", x.shape,  self.image_size, self.patch_size, self.num_patches)
        x = x.flatten(2).transpose(1, 2)
        return x



class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        self.position_embeddings = \
        nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches+1, config["hidden_size"]))
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # print("Token: ", cls_tokens.shape)
        # print("X: ", x.shape)
        x = torch.cat((cls_tokens, x), dim=1)
        # print("X: ", x.shape)
        # print("self.position_embeddings: ", self.position_embeddings.shape)
        x = x+self.position_embeddings
        x = self.dropout(x)
        return x



class AttentionHead(nn.Module):
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        query = self.query(x)
        key = self.query(x)
        value = self.value(x)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        attention_output = torch.matmul(attention_probs, value)
        return attention_output, attention_probs





class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super.__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]

        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.qkv_bias = config["qkv_bias"]
        self.heads = nn.ModuleList([])

        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias
            )
            self.heads.append(head)
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        attention_outpus = [head(x) for head in self.heads]
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outpus], dim=-1)
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        if not output_attentions:
            return (attention_output, None)
        else:
            attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outpus], dim=-1)

            return (attention_output, attention_probs)


class FasterMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.qkv_bias = config["qkv_bias"]
        self.qkv_projection = nn.Linear(self.hidden_size, self.all_head_size*3, bias=self.qkv_bias)
        self.attn_dropout = nn.Dropout(config["attention_probs_dropout_prob"])
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])
        

    def forward(self, x, output_attentions=False):

        qkv = self.qkv_projection(x)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        batch_size, sequence_length, _ = query.size()
        # Resize the query, key, and value to (batch_size, num_attention_heads, sequence_length, attention_head_size)
        
        query = query.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1,2)
        key = key.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        
        #Calculate the attention scores
        # softmax (Q*K.T / sqrt(head_size)) *V

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)

        attention_ouput = torch.matmul(attention_probs, value)
        attention_ouput = attention_ouput.transpose(1, 2) \
                            .contiguous().view(batch_size, sequence_length, self.all_head_size)
        
        attention_output = self.output_projection(attention_ouput)
        attention_ouput = self.output_dropout(attention_ouput)
        if  not output_attentions:
            return (attention_ouput, None)
        else:
            return (attention_ouput, attention_probs)




class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = GeluActivation()
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)

        return x




class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_faster_attention = config.get("use_faster_attention", False)
        if self.use_faster_attention:
            self.attention = FasterMultiHeadAttention(config)
        else:
            self.attention = MultiHeadAttention(config)
        
        self.layernorm_1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x, output_attentions=False):
        attention_output, attention_probs = \
        self.attention(self.layernorm_1(x), output_attentions=output_attentions)
        x = x+attention_output
        mlp_output = self.mlp(self.layernorm_2(x))
        x = x+mlp_output
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)




class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions = output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)


class ViTForClassfication(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]
        self.embedding = Embeddings(config)
        self.encoder = Encoder(config)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
        # print(x.shape)
        embedding_output = self.embedding(x)
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
        logits = self.classifier(encoder_output[:,0,:])
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if  module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"]
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std = self.config["initializer_range"]
            ).to(module.cls_token.dtype)






cv = LeaveOneOut()
pathname_reaction = '/scratch/dazhang/MATLAB/'



def save_experiment(experiment_name, config, model, train_losses, test_losses, accuracies, base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)

    # Save the config
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)

    # Save the metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'w') as f:
        data = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'accuracies': accuracies,
        }
        json.dump(data, f, sort_keys=True, indent=4)

    # Save the model
    save_checkpoint(experiment_name, model, "final", base_dir=base_dir)


def save_checkpoint(experiment_name, model, epoch, base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f'model_{epoch}.pt')
    torch.save(model.state_dict(), cpfile)


def load_experiment(experiment_name, checkpoint_name="model_final.pt", base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    # Load the config
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'r') as f:
        config = json.load(f)
    # Load the metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    train_losses = data['train_losses']
    test_losses = data['test_losses']
    accuracies = data['accuracies']
    # Load the model
    model = ViTForClassfication(config)
    # model = nn.DataParallel(model)
    cpfile = os.path.join(outdir, checkpoint_name)
    
    model.load_state_dict(torch.load(cpfile))
    # model = model.module.to(device)
    return config, model, train_losses, test_losses, accuracies



train_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((config['image_size'], config['image_size']), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop((config['image_size'], config['image_size']), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
        transforms.Normalize((0.5), (0.5))
        ])
    

test_transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Resize((config['image_size'], config['image_size']), antialias=True),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


batch_size=256
num_workers=2




import sklearn.metrics as metrics

def cal_auc_roc(y_true, y_pred):
    fpr_cnn, tpr_cnn, _ = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.roc_auc_score(y_true, y_pred)
    roc_auc = round(roc_auc,5)
    precision1, recall1, thresholds = precision_recall_curve(y_true, y_pred)
    prauc = round(auc(recall1, precision1),6)

    return roc_auc, prauc

def calc_precision_recall(y_true, y_pred): 
    TP = 0
    FP = 0
    FN = 0

    for i in range(len(y_true)): 
        if y_true[i]==y_pred[i]==1:
           TP += 1
        if y_pred[i]==1 and y_true[i]!=y_pred[i]:
           FP += 1
        if y_pred[i]==0 and y_true[i]!=y_pred[i]:
           FN += 1
    
    try:
        precision = TP / (TP + FP)
    except:
        precision = 1
    
    try:
        recall = TP / (TP + FN)
    except:
        recall = 1

    return precision, recall


import torch
import torchvision
import torchvision.transforms as transforms


save_image = []
save_attention = []
gts = []
preds = []

class Trainer:
    """
    The simple trainer.
    """

    def __init__(self, model, optimizer, loss_fn, exp_name, config, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device
        self.best_model = None


    @torch.no_grad()
    def plot_testloader(self, testloader):

        """
        Visualize the attention maps of the first 4 images.
        """
        plt.ioff()
        self.best_model.eval()
        classes = ('ripple', 'spike')
        save_attention = []
        save_image = []
        

        # with torch.no_grad():
        cur_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        self.best_model = self.best_model.to(cur_device)
        for batch in testloader:
            
            images, labels = batch[0].to(cur_device), batch[1].to(cur_device) # (256  "This represents Batch Size  !!!!!!!")
            
            # Get predictions
            logits, attention_maps = self.best_model(images, output_attentions=True)
            # print(logits.shape, len(attention_maps), attention_maps[0].shape)

            predictions = torch.argmax(logits, dim=1)
            attention_maps = torch.cat(attention_maps, dim=1)
            attention_maps = attention_maps[:, :, 0, 1:]
            attention_maps = attention_maps.mean(dim=1)
            num_patches = attention_maps.size(-1)
            size = int(math.sqrt(num_patches))
            attention_maps = attention_maps.view(-1, size, size)
            attention_maps = attention_maps.unsqueeze(1)
            attention_maps = F.interpolate(attention_maps, size=(config['image_size']+1, config['image_size']+1), mode='bilinear', align_corners=False)
            attention_maps = attention_maps.squeeze(1)

            fig = plt.figure(figsize=(35, 20))
            mask = np.concatenate([np.ones((config['image_size']+1, config['image_size']+1)), np.zeros((config['image_size']+1, config['image_size']+1))], axis=1)
            for i in range(min(1, 1)):
                ax = fig.add_subplot(7, 7, i+1, xticks=[], yticks=[])
                img = np.concatenate((images[i].cpu(), images[i].cpu()), axis=1)
                # print(img.shape)
                ax.imshow(img[0])
                save_image.append(img[0])
                # Mask out the attention map of the left image
                extended_attention_map = np.concatenate((np.zeros((config['image_size']+1, config['image_size']+1)), attention_maps[i].cpu()), axis=1)
                extended_attention_map = np.ma.masked_where(mask==1, extended_attention_map)
                save_attention.append(extended_attention_map)
                ax.imshow(extended_attention_map, alpha=0.5, cmap='jet')
                # Show the ground truth and the prediction
                gt = classes[labels[i]]
                gts.append(labels[i])
                pred = classes[predictions[i]]
                preds.append(predictions[i])
                ax.set_title(f"gt: {gt} / pred: {pred}", color=("green" if gt==pred else "red"))
            # if output is not None:
                # plt.savefig("only_animal.png")
            # plt.show()
        return img[0], extended_attention_map

    @torch.no_grad()
    def plot_batchloader(self, testloader):

        """
        Visualize the attention maps of the first 4 images.
        """
        plt.ioff()
        self.best_model.eval()
        classes = ('ripple', 'spike')
        save_attention = []
        save_image = []
        
        

        # with torch.no_grad():
        cur_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        self.best_model = self.best_model.to(cur_device)
        ripple_save_attention = []
        ripple_save_image = []
        spike_save_attention = []
        spike_save_image = []

        for batch in testloader:
            
            images, labels = batch[0].to(cur_device), batch[1].to(cur_device) # (256  "This represents Batch Size  !!!!!!!")
            
            # Get predictions
            logits, attention_maps = self.best_model(images, output_attentions=True)
            

            predictions = torch.argmax(logits, dim=1)
            attention_maps = torch.cat(attention_maps, dim=1)
            attention_maps = attention_maps[:, :, 0, 1:]
            attention_maps = attention_maps.mean(dim=1)
            num_patches = attention_maps.size(-1)
            size = int(math.sqrt(num_patches))
            attention_maps = attention_maps.view(-1, size, size)
            attention_maps = attention_maps.unsqueeze(1)
            attention_maps = F.interpolate(attention_maps, size=(config['image_size']+1, config['image_size']+1), mode='bilinear', align_corners=False)
            attention_maps = attention_maps.squeeze(1)
            print((attention_maps.shape),  "This represents Batch Size  !!!!!!!")

            fig = plt.figure(figsize=(35, 20))
            mask = np.concatenate([np.ones((config['image_size']+1, config['image_size']+1)), np.zeros((config['image_size']+1, config['image_size']+1))], axis=1)

            # for i in range(1):
            for i in range(len(images)):
                ax = fig.add_subplot(18, 18, i+1, xticks=[], yticks=[])
                img = np.concatenate((images[i].cpu(), images[i].cpu()), axis=1)
                # print(img.shape)
                ax.imshow(img[0])
                save_image.append(img[0])
                # Mask out the attention map of the left image
                extended_attention_map = np.concatenate((np.zeros((config['image_size']+1, config['image_size']+1)), attention_maps[i].cpu()), axis=1)
                extended_attention_map = np.ma.masked_where(mask==1, extended_attention_map)
                save_attention.append(extended_attention_map)
                ax.imshow(extended_attention_map, alpha=0.5, cmap='jet')
                # Show the ground truth and the prediction
                gt = classes[labels[i]]
                gts.append(labels[i])
                pred = classes[predictions[i]]
                # print(gt, pred)
                if (gt == pred) and (gt=="ripple"):
                    ripple_save_attention.append(extended_attention_map)
                elif (gt == pred) and (gt=="spike"):
                    spike_save_attention.append(extended_attention_map)

                preds.append(predictions[i])
                ax.set_title(f"gt: {gt} / pred: {pred}", color=("green" if gt==pred else "red"))
            
        print(len(ripple_save_attention), "True Ripples!!!!")
        print(len(spike_save_attention), "True Spikes!!!!")

        avg_pos_img = np.mean(np.asarray(spike_save_attention)[:,:,:], axis=0, dtype=None, out=None)
        # avg_pos_img = np.mean(np.asarray(ripple_save_attention)[:,:,:], axis=0, dtype=None, out=None)
        plt.figure(figsize=(35, 20))
        axes = plt.axes()
        plt.imshow(avg_pos_img[:,config['image_size']+1:], cmap='jet', extent=[0, 103, 1, 200], aspect=0.565)
        savemat('avg_spike_attention.mat', {'array_name': avg_pos_img[:,config['image_size']+1:]})
        # savemat('avg_ripple_attention.mat', {'array_name': avg_pos_img[:,config['image_size']+1:]})
        # plt.imshow(avg_pos_img[:,config['image_size']+1:], cmap='jet', extent=[0,512,1,200], aspect=8.393)
        plt.colorbar()

        annots = loadmat('freq_61.mat')
        freq_val = annots['freq_61']

        freq_val = list(freq_val[0])
        freq_val.append(200.00)
        freq_val.insert(0, 1.00)
        freq_val = [round(f, 1) for f in freq_val]


        plt.xlim(0, 103)
        xticks = [i for i in range(0, 103, 32)]
        ticklabels = [str(i) for i in range(0, 103, 32)]
        plt.xticks(xticks, ticklabels)

        plt.ylim(1, 200)
        yticks = freq_val

        ticklabels = [str(i) for i in freq_val]
        # ticklabels = [1,4,10,20,50,100,200]
        plt.yticks(yticks, ticklabels)
        # axes.set_yscale('log', base=1.09)
        # axes.yaxis.set_minor_formatter(mticker.ScalarFormatter())
        # axes.set_yticks(ticklabels)

        plt.savefig("ttttttttttttttt_true_label_200.png")
        print((avg_pos_img.shape), "????????????!!!!")

        return img[0], extended_attention_map, avg_pos_img[:,config['image_size']+1:]

    
    def calculate_metrics(self, y_true, y_pred):

        # print("As in Numpy Array")
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()

        
        # y_pred = np.amax(y_pred, axis=1)
        y_pred = y_pred[:, 1]
        # print(len(y_pred))
        # print(y_true)

        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
        roc_auc = metrics.roc_auc_score(y_true, y_pred)

    # def train(self, trainloader, testloader, epochs, testing_id, save_model_every_n_epochs=0):
    def train(self, trainloader, testloader, epochs, save_model_every_n_epochs=0):
        """
        Train the model for the specified number of epochs.
        """
        # Keep track of the losses and accuracies
        train_losses, test_losses, accuracies = [], [], []
        accuracy_list = [0]
        predictions_list = [0]
        best_loss = float('inf')
        # Train the model
        for i in range(epochs):
            train_loss, train_accuracy, trial_accuracy = self.train_epoch(trainloader)
            accuracy, predictions, test_loss, trial_accuracy_valid, tensor_trial_probs, tensor_trial_label, cpu_l, spike_prob \
            = self.evaluate(testloader)

            if i%2 == 0:
                self.calculate_metrics(tensor_trial_label, tensor_trial_probs)

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            print(f"Epoch: {i+1}, Train loss: {train_loss:.4f}, Train ACC: {train_accuracy:.4f}, | Test loss: {test_loss:.4f}, Test ACC.: {accuracy:.4f} ")
            
            if train_loss<best_loss:
                self.best_model = self.model

            if i == epochs - 1:
                # save_img, save_att = self.plot_testloader(testloader)
                save_img, save_att, avg_pos_img = self.plot_batchloader(testloader)

                
        # Save the experiment
        save_experiment(self.exp_name+"_last", config, self.model, train_losses, test_losses, accuracies)
        save_experiment(self.exp_name+"_best", config, self.best_model, train_losses, test_losses, accuracies)
        # return accuracy_list, predictions_list, save_img, save_att
        return accuracy_list, predictions_list, save_img, save_att, avg_pos_img, cpu_l, spike_prob
        

    def train_epoch(self, trainloader):
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_loss = 0
        train_correct = 0
        trial_correct = 0
        for batch in trainloader:
            # Move the batch to the device
            # batch = [t.to(self.device) for t in batch]
            images, labels, trial_label = batch[0].to(self.device), batch[1].to(self.device),  batch[2].to(self.device)
            # Zero the gradients
            self.optimizer.zero_grad()
            # Calculate the loss
            train_logits, _ = self.model(images)
            loss = self.loss_fn(self.model(images)[0], labels)
            train_predictions = torch.argmax(train_logits, dim=1)
            train_correct += torch.sum(train_predictions == labels).item()
            trial_correct += torch.sum(train_predictions == trial_label).item()
            # Backpropagate the loss
            loss.backward()
            # Update the model's parameters
            self.optimizer.step()
            total_loss += loss.item() * len(images)
            # torch.cuda.empty_cache()
        train_accuracy = train_correct / len(trainloader.dataset)
        trial_accuracy = trial_correct / len(trainloader.dataset)
        # print(total_loss / len(trainloader.dataset))
        return total_loss / len(trainloader.dataset), train_accuracy, trial_accuracy

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        trial_correct = 0
        tensor_trial_probs = torch.empty((0, 2)).to(self.device)
        tensor_trial_label = torch.tensor([]).to(self.device)

        cpu_pred = list()
        cpu_l = list()
        cpu_prediction_label = list()
        spike_prob = list()

        with torch.no_grad():
            for batch in testloader:
                # Move the batch to the device
                batch = [t.to(self.device) for t in batch]
                
                images, labels, trial_label = batch
                
                # Get predictions
                logits, _ = self.model(images)

                # Calculate the loss
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(images)

                # Calculate the accuracy
                # predictions = torch.argmax(logits, dim=1)
                # predictions_value = torch.amax(torch.sigmoid(logits), dim=1)
                
                # cpu_prediction_value = predictions_value.cpu().numpy()
                # # cpu_prediction = torch.sigmoid(logits).cpu().numpy()
                # cpu_labels = labels.cpu().numpy()
                # cpu_pred.extend(cpu_prediction_value)
                # cpu_l.extend(cpu_labels)
                # cpu_prediction_label.extend(predictions.cpu().numpy())
                # # print("Logits......", torch.sigmoid(logits) )

                # tensor_trial_probs = torch.cat((tensor_trial_probs, torch.sigmoid(logits)))
                # tensor_trial_label = torch.cat((tensor_trial_label, trial_label))


                # Softmax probabilities for both classes
                probabilities = torch.softmax(logits, dim=1)
                # Predicted class
                predictions = torch.argmax(probabilities, dim=1)
                # Probability of class 1 (Spike / IED)
                spike_prob_batch = probabilities[:, 1]
                cpu_l.extend(labels.cpu().numpy())
                cpu_prediction_label.extend(predictions.cpu().numpy())
                spike_prob.extend(spike_prob_batch.cpu().numpy().tolist())

                tensor_trial_probs = torch.cat((tensor_trial_probs, probabilities), dim=0)
                tensor_trial_label = torch.cat((tensor_trial_label, trial_label), dim=0)
                # print("Logits......",  tensor_trial_probs.shape, tensor_trial_label.shape)  

                correct += torch.sum(predictions == labels).item()
                trial_correct += torch.sum(predictions == trial_label).item()
        accuracy = correct / len(testloader.dataset)
        trial_accuracy = trial_correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)

        # for i, pred_label in enumerate(cpu_prediction_label):
        #     if pred_label == 1:
        #         spike_prob.append(cpu_pred[i])
        #     else:
        #         spike_prob.append(1-cpu_pred[i])
        
        print("Prediction label: ", cpu_prediction_label[30:70])
        print("True       label: ",cpu_l[30:70])
        print("Predicted Prob: ",spike_prob[30:70])

        print("Groundtruth positive: ", sum(cpu_l[i] for i in range(len(cpu_l)) if cpu_l[i] == 1))
        print("Groundtruth negative: ", sum((1-cpu_l[i]) for i in range(len(cpu_l)) if cpu_l[i] == 0))

        print("True positive: ", sum(cpu_prediction_label[i] == cpu_l[i] for i in range(len(cpu_prediction_label)) if cpu_l[i] == 1))
        print("True negative: ", sum(cpu_prediction_label[i] == cpu_l[i] for i in range(len(cpu_prediction_label)) if cpu_l[i] == 0))
        print("False positive: ", sum(cpu_prediction_label[i] != cpu_l[i] for i in range(len(cpu_prediction_label)) if cpu_prediction_label[i] == 1))
        print("False negative: ", sum(cpu_prediction_label[i] != cpu_l[i] for i in range(len(cpu_prediction_label)) if cpu_prediction_label[i] == 0))
        

        precision1, recall1, thresholds = precision_recall_curve(cpu_l, spike_prob)
        auc_precision_recall = round(auc(recall1, precision1),5)
        # lr_auc = roc_auc_score(cpu_l, cpu_pred)
        lr_auc = roc_auc_score(cpu_l, spike_prob)
        fpr, tpr, _ = roc_curve(cpu_l, spike_prob)
        
        plt.figure()
        plt.plot(fpr, tpr, marker='.', label='AUC-ROC Score:(area = {:.3f})'.format(lr_auc))
        plt.plot([0, 1], [0, 1], 'k-')
        # plt.plot(recall1, precision1, marker='.', label='Precision-Recall Score:(area = {:.3f})'.format(auc_precision_recall))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.ylim(0.0,1.0)
        plt.xlim(0.0,1.0)
        # show the legend
        plt.legend()
        # show the plot
        plt.show()
        
        plt.savefig("pppppppppp_200.png")
        print(lr_auc, auc_precision_recall,  "@@@@@@@@@@@@@@@@@@@@@@")

        # cpu_l = [1-i for i in cpu_l]
        
        return accuracy, predictions, avg_loss, trial_accuracy, tensor_trial_probs, tensor_trial_label, cpu_l, spike_prob


@torch.no_grad()
def visualize_attention(model, output, testset, config, device="cuda"):
    """
    Visualize the attention maps of the first 4 images.
    """
    model.eval()
    # Load random images
    num_images = 49
    
    classes = ('ripple', 'spike')
    # Pick 30 samples randomly
    indices = torch.randperm(len(testset))[:num_images]

    # raw_images = [np.asarray(testset[i][0]) for i in indices]
    raw_images = [testset[i][0] for i in indices]
    labels = [testset[i][1] for i in indices]

    images = torch.stack([image for image in raw_images])
    # Move the images to the device
    images = images.to(device)
    model = model.to(device)
    # Get the attention maps from the last block
    logits, attention_maps = model(images, output_attentions=True)
    # Get the predictions
    predictions = torch.argmax(logits, dim=1)
    # Concatenate the attention maps from all blocks
    attention_maps = torch.cat(attention_maps, dim=1)
    # select only the attention maps of the CLS token
    attention_maps = attention_maps[:, :, 0, 1:]
    # Then average the attention maps of the CLS token over all the heads
    attention_maps = attention_maps.mean(dim=1)
    # Reshape the attention maps to a square
    num_patches = attention_maps.size(-1)
    size = int(math.sqrt(num_patches))
    attention_maps = attention_maps.view(-1, size, size)
    # Resize the map to the size of the image
    attention_maps = attention_maps.unsqueeze(1)
    attention_maps = F.interpolate(attention_maps, size=(config['image_size'], config['image_size']), mode='bilinear', align_corners=False)
    attention_maps = attention_maps.squeeze(1)
    # Plot the images and the attention maps
    fig = plt.figure(figsize=(20, 10))
    mask = np.concatenate([np.ones((config['image_size'], config['image_size'])), np.zeros((config['image_size'], config['image_size']))], axis=1)
    for i in range(num_images):
        ax = fig.add_subplot(7, 7, i+1, xticks=[], yticks=[])
        img = np.concatenate((raw_images[i], raw_images[i]), axis=1)
        ax.imshow(img[0])
        # Mask out the attention map of the left image
        extended_attention_map = np.concatenate((np.zeros((config['image_size'], config['image_size'])), attention_maps[i].cpu()), axis=1)
        extended_attention_map = np.ma.masked_where(mask==1, extended_attention_map)
        ax.imshow(extended_attention_map, alpha=0.5, cmap='jet')
        # Show the ground truth and the prediction
        gt = classes[labels[i]]
        pred = classes[predictions[i]]
        ax.set_title(f"gt: {gt} / pred: {pred}", color=("green" if gt==pred else "red"))
    if output is not None:
        plt.savefig(output)
    plt.show()


def main():
    # Training parameters

    all_specs = np.empty((0, 61*103))   # 513 (timestamps) x 61 (frequency)
    file_list = []
    
    config = {
        "patch_size": 4,  # Input image size: 64x64 -> 8x8 patches
        "hidden_size": 12,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "intermediate_size": 4 * 12, # 4 * hidden_size
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "initializer_range": 0.02,
        "image_size": 32,
        "num_classes": 2, # num_classes of CIFAR10
        "num_channels": 1,
        "qkv_bias": True,
        "use_faster_attention": True,
    }
    # These are not hard constraints, but are used to prevent misconfigurations
    assert config["hidden_size"] % config["num_attention_heads"] == 0
    assert config['intermediate_size'] == 4 * config['hidden_size']
    assert config['image_size'] % config['patch_size'] == 0


    # for idx, subject in enumerate(sub_list):
    # print( subject)
    import sys
    path = '/data/an/spectrograms_other_data/'
    patient = sys.argv[1]
    file_path = os.path.join(path, f"{patient}_ls_spec_negative.mat")
    # print(path)

    ls_specs = np.empty((0, 61*103))


    # for filename in glob.glob(os.path.join(path, '*.mat')):
    #     print(filename)
    #     file_path = os.path.join(os.getcwd(), filename)
        
    with h5py.File(file_path, "r") as f:
        specs = np.asarray(f["ls_neg_spec"]).T
        print(specs.shape)
        # specs = loadmat(filename)['ls_ripple']
        # cur_channel_list.append(specs.shape[0])
        ls_specs = np.concatenate([ls_specs, specs], axis=0)

    print(ls_specs.shape)

    x_test = np.concatenate([ls_specs])
    y_test = np.concatenate([[0]*ls_specs.shape[0]])
    trial_ref_test = np.concatenate([[0]*ls_specs.shape[0]])
       

    testset = RippleTrialDataset_200(x_test, y_test, trial_ref_test, transformer=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False,  pin_memory=True, num_workers=num_workers)

    print(testloader)
    model = ViTForClassfication(config)


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    # exp_name = "ripple_spike_200_"+str(4)+"_true_label"
    exp_name = "ripple_spike_200_"+str(0)+"cana_leave_"
    # exp_name = "ripple_spike_200_"+str(0)
    config, model, train_losses, test_losses, accuracies = load_experiment(exp_name+"_best/")

    cur_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    best_model = model.to(cur_device)
    save_dir = "./pred_not_0_figures"
    os.makedirs(save_dir, exist_ok=True)
    global_idx = 0 
    saved_count = 0
    all_sample_indices = []
    all_true_labels = []
    all_pred_labels = []
    all_ied_probs = []
    all_bad_indices = []
   
    from scipy.io import loadmat
    from matplotlib.colors import ListedColormap

    mat = loadmat('parula_colormap.mat')
    parula = ListedColormap(mat["cmap"])
    
    annots = loadmat('freq_61.mat')
    frx = annots['freq_61']

    for batch_idx, batch in enumerate(testloader): 
        images, labels = batch[0].to(cur_device), batch[1].to(cur_device) # (256  "This represents Batch Size  !!!!!!!")
        logits, attention_maps = best_model(images, output_attentions=True)
        predictions = torch.argmax(logits, dim=1)
        probabilities = torch.softmax(logits, dim=1)
        print(predictions)
        # print(probabilities)
        tensor_sum = predictions.sum().item()
        print(tensor_sum)
        bad_indices = (predictions != 0).nonzero(as_tuple=True)[0]
        for i in bad_indices.tolist():
            sample_idx = global_idx + i
            # -----------------------------------------
            # Recover original MATLAB spectrogram
            # ls_spec row: 6283 = 61 * 103
            # -----------------------------------------
            spectrogram = x_test[sample_idx].reshape(61, 103, order='F')
            pred = predictions[i].item()
            prob = probabilities[i].detach().cpu().numpy()
            # -----------------------------------------
            # MATLAB:
            #
            # figure;
            # pcolor(x, frx, spectrogram)
            # shading flat
            # set(gca,'YScale','log')
            # caxis([-2 2])
            # -----------------------------------------

            fig, ax = plt.subplots(figsize=(7, 5))
            # 103 samples centered around event
            x = np.arange(-51, 52)
            frx_plot = np.asarray(frx).squeeze()
            time_axis = np.arange(-51, 52) / 512.0
            print("spectrogram:", spectrogram.shape)
            # print("time_axis:", time_axis.shape)
            print("frx:", np.asarray(frx).shape)

            pcm = ax.pcolormesh(
                time_axis,
                frx_plot,
                spectrogram,
                shading="auto",
                cmap=parula,
                vmin=-2,
                vmax=2
            )

            ax.set_yscale("log")
            ax.set_xlabel("Samples relative to event")
            ax.set_ylabel("Frequency (Hz)")

            ax.set_title(
                f"{patient} | trial {sample_idx} | "
                f"pred={pred} | "
                f"P(0)={prob[0]:.3f}, P(1)={prob[1]:.3f}"
            )

            fig.colorbar(pcm,ax=ax,label="Z-scored amplitude")

            fig.tight_layout()

            save_file = os.path.join(
                save_dir,
                f"{patient}_trial_{sample_idx:04d}_pred_{pred}.png"
            )

            fig.savefig(
                save_file,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close(fig)

            print("Saved:", save_file)

            saved_count += 1

        # important: update after each batch
        global_idx += images.shape[0]
    return 

    print("All raw spectrograms: ", all_specs.shape)
    

    tr_label = list()
    manual_label_spikes = list()
    trial_time = list()
    
        
    
    print(len(manual_label_spikes), "manual_label_spikes")
     

    all_tr = []
    for i, t in enumerate(tr_label):
        if math.isnan(t[0]) or t[0]==0:
            all_tr.append(0)
        else:
            all_tr.append(1)

    all_tr = np.asarray(all_tr)
    all_tr = all_tr.reshape((len(all_tr),))
    print("Trial lables: ",all_tr.shape)
    print("Correct Trials VS Incorrect Trials *******", Counter(all_tr))

    all_manual_spikes = []
    for i, t in enumerate(manual_label_spikes):
        if math.isnan(t[0]) or t[0]==0:
            all_manual_spikes.append(0)
        else:
            all_manual_spikes.append(1)

    all_manual_spikes = np.asarray(all_manual_spikes)
    all_manual_spikes = all_manual_spikes.reshape((len(all_manual_spikes),))
    print("Manual Spike lables: ",all_manual_spikes.shape)
    print("Manual Label Non Spikes VS Spikes ###### ", Counter(list(all_manual_spikes)))
     



    ripple_label_spike_labels = [i for i in manual_label_spikes]
    all_labels = np.array(ripple_label_spike_labels)
    
    
    cluster_spike = 1  # spike
    cluster_ripple = 0  # ripple
    
    item_index_spike = np.where(all_labels==cluster_spike)
    item_index_ripple = np.where(all_labels==cluster_ripple)
    # print(len(item_index_0), item_index_0[0], item_index_0[1])

    x_cluster_spike = all_specs[item_index_spike[0]] 
    x_cluster_ripple = all_specs[item_index_ripple[0]]


    trial_cluster_spike = all_tr[item_index_spike[0]]
    trial_cluster_ripple = all_tr[item_index_ripple[0]]
    
    print("Spike Sample Size: ", x_cluster_spike.shape)
    print("Ripple Sample Size: ", x_cluster_ripple.shape)

    x = np.concatenate([x_cluster_spike, x_cluster_ripple])
    y = np.concatenate([[1]*x_cluster_spike.shape[0], [0]*x_cluster_ripple.shape[0]])
    trials = np.concatenate([trial_cluster_spike, trial_cluster_ripple])

    
    idx = np.random.permutation(len(x))
    x,y,tt_label = x[idx], y[idx], trials[idx]


    config = {
    "patch_size": 4,  # Input image size: 64x64 -> 8x8 patches
    "hidden_size": 12,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "intermediate_size": 4 * 12, # 4 * hidden_size
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 2, # num_classes of CIFAR10
    "num_channels": 1,
    "qkv_bias": True,
    "use_faster_attention": True,
    }
    # These are not hard constraints, but are used to prevent misconfigurations
    assert config["hidden_size"] % config["num_attention_heads"] == 0
    assert config['intermediate_size'] == 4 * config['hidden_size']
    assert config['image_size'] % config['patch_size'] == 0


    
    

    final = []
    # idd = 0
    # for train_ix, test_ix in cv.split(x):
    from sklearn.model_selection import KFold
    # k = 10
    k = 5
    ripple_avg_attention = []

    # splits = KFold(n_splits=k,random_state=None,shuffle=True)
    # for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(x)))):
    skf = StratifiedKFold(n_splits=k,random_state=None,shuffle=True)
    for fold, (train_idx, val_idx) in enumerate(skf.split(x, y)):
        epochs = 200
        # epochs = 200 # 80 could be potential number
        # if fold == 4 or (fold ==2):
        #     epochs = 30
        # if fold == 3:
        #     epochs = 30
        save_attention = []

        print("This is for the: %s fold"%(fold+1)+ " Train Size: "+str(len(train_idx))+ " Test Size: "+str(len(val_idx)))
        
        # count += 1
        x_train, x_test = x[train_idx, :], x[val_idx, :]
        y_train, y_test = y[train_idx], y[val_idx]
        trial_ref_train, trial_ref_test  = tt_label[train_idx], tt_label[val_idx]


        trainset = RippleTrialDataset_200(x_train, y_train, trial_ref_train, transformer=train_transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, pin_memory=True, num_workers=num_workers)

        testset = RippleTrialDataset_200(x_test, y_test, trial_ref_test, transformer=test_transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False,  pin_memory=True, num_workers=num_workers)


        
        save_model_every_n_epochs = save_model_every
        # Create the model, optimizer, loss function and trainer
        model = ViTForClassfication(config)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)

        # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-1)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1) #for Channel 38
        loss_fn = nn.CrossEntropyLoss()
        exp_name = "ripple_spike_200_"+str(fold)+"_true_label"
        trainer = Trainer(model, optimizer, loss_fn, exp_name, config, device=device)
        # accuracy_list, predictions_list, save_img, save_att =  trainer.train(trainloader, testloader, epochs, save_model_every_n_epochs=save_model_every_n_epochs)
        accuracy_list, predictions_list, save_img, save_att, avg_pos_img, cpu_l, cpu_pred = trainer.train(trainloader, testloader, epochs, save_model_every_n_epochs=save_model_every_n_epochs)
        # visualize_images(trainset,"test_ripple_"+str(fold))
        save_image.append(save_img)
        save_attention.append(save_att)

        ripple_avg_attention.append(avg_pos_img)
        

        file1 = open("true_label_file_200.txt", "a")  # append mode
        file1.write(''.join(str(cpu_l))+"\n")
        file1.close()
        file2 = open("true_pred_file_200.txt", "a")  # append mode
        file2.write(''.join(str(cpu_pred))+"\n")
        file2.close()
        
        config, model, train_losses, test_losses, accuracies = load_experiment(exp_name+"_best/")
        visualize_attention(model,"test_ripple_attention_"+str(fold), trainset, config)

        print("===========================================================")
        print("===========================================================")
        print("===========================================================")

    ripple_avg_attention = np.mean(np.asarray(ripple_avg_attention)[:,:,:], axis=0, dtype=None, out=None)
    plt.figure(figsize=(35, 20))
    axes = plt.axes()
    savemat('avg_spike_attention_all.mat', {'array_name': ripple_avg_attention})
    # savemat('avg_ripple_attention_all.mat', {'array_name': ripple_avg_attention})
    plt.imshow(ripple_avg_attention, cmap='jet', extent=[0,102,1,200], aspect=0.565)
    # plt.imshow(ripple_avg_attention, cmap='jet', extent=[0,512,1,200], aspect=8.393)
    

    annots = loadmat('freq_61.mat')
    freq_val = annots['freq_61']

    freq_val = list(freq_val[0])
    freq_val.append(200.00)
    freq_val.insert(0, 1.00)
    freq_val = [round(f, 1) for f in freq_val]


    plt.xlim(0, 102)
    xticks = [i for i in range(0, 103, 32)]
    ticklabels = [str(i) for i in range(0, 103, 32)]
    plt.xticks(xticks, ticklabels)

    plt.ylim(1, 200)
    yticks = freq_val
    ticklabels = [str(i) for i in freq_val]
    # ticklabels = [1,4,10,20,50,100,200]
    plt.yticks(yticks, ticklabels)
    # axes.set_yscale('log', base=1.09)
    # axes.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    # axes.set_yticks(ticklabels)
    plt.colorbar()
    plt.savefig("ssssssssssss_true_label_200.png")
    
    print((avg_pos_img.shape), "????????????!!!!")



if __name__ == '__main__':
    main()