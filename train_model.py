#!/usr/bin/env python
# coding: utf-8
import argparse
import torch
import numpy as np
import os
import os.path as osp
import sys
import time
from tqdm.autonotebook import tqdm
import matplotlib.pylab as plt
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

sys.path.append('.')

## Imports based on our ready-to-use code (after you pip-install the cs233_gtda_hw4 package)
from cs233_gtda_hw4.in_out.utils import make_data_loaders
from cs233_gtda_hw4.in_out.utils import save_state_dicts, load_state_dicts
from cs233_gtda_hw4.in_out import pointcloud_dataset
from cs233_gtda_hw4.in_out.plotting import plot_3d_point_cloud

from cs233_gtda_hw4.losses import EncodingDistance

## Imports you might use if you follow are scaffold code (it is OK to use your own stucture of the models)
from cs233_gtda_hw4.models import FancyPartAwarePointcloudAutoencoder
from cs233_gtda_hw4.models.point_net import PointNet
from cs233_gtda_hw4.models.mlp import MLP


# In[2]:


##
## Fixed Settings (we do not expect you to change these)
## 

n_points = 1024  # number of points of each point-cloud
n_parts = 4      # max number of parts of each shape

# batch-size of data loaders
batch_size = 128 # if you can keep this too as is keep it, 
                 # but if it is too big for your GPU, feel free to change it.

device = 'cuda' if torch.cuda.is_available() else 'cpu'

top_in_dir = './cs233_gtda_hw4/data/'
top_out_dir = './cs233_gtda_hw4/data/out/'
if not osp.exists(top_out_dir):
    os.makedirs(top_out_dir)
    
top_log_dir = './cs233_gtda_hw4/data/logs/'
if not osp.exists(top_log_dir):
    os.makedirs(top_log_dir)

# PREPARE DATA:
loaders = make_data_loaders(top_in_dir, batch_size)

for split, loader in loaders.items():
    print('N-examples', split, len(loader.dataset))
    
# PARSE ARGS AND BUILD MODELS:
parser = argparse.ArgumentParser()
parser.add_argument('--out_n', type=int, default=1024)
parser.add_argument('--drop', type=float, default=0)
parser.add_argument('--bnorm', action='store_true', default=False)
parser.add_argument('--noise', type=float, default=0)
parser.add_argument('--alpha', type=float, default=0, help='regularizer for decoder')
parser.add_argument('--cdec', type=float, default=1, help='decay for point classifier lambda')
parser.add_argument('--part_lambda', type=float, default=5e-3, help='point classifier lambda')
parser.add_argument('--init_lr', type=float, default=9e-3, help='initial ADAM LR')
parser.add_argument('--exist_lambda', type=float, default=5e-3, help='part existence lambda')
parser.add_argument('--kl_lambda', type=float, default=1e-5, help='KL divergence lambda')
parser.add_argument('--kdec', type=float, default=1, help='decay (or growth) for KL lambda')
parser.add_argument('--encode_parts', action='store_true', default=False)
parser.add_argument('--predict_parts', action='store_true', default=True)
parser.add_argument('--predict_part_exist', action='store_true', default=False)
parser.add_argument('--penal_parts', type=float, default=0, help='penalize reconstructed non-parts')
parser.add_argument('--variational', action='store_true', default=False)
parser.add_argument('--hdim', type=int, default=128, help='Dimension of latent space')
parser.add_argument('--n_epochs', type=int, default=500, help='epochs to train for')
args = parser.parse_args()

out_n = args.out_n
drop = args.drop
bnorm = args.bnorm
noise = args.noise
alpha = args.alpha
cdec = args.cdec
part_lambda = args.part_lambda # for the part-aware AE you will be using (summing) two losses:
                     # chamfer + cross-entropy
                     # do it like this: chamfer + (part_lambda * cross-entropy), 
                     # i.e. we are scaling down the cross-entropy term
init_lr = args.init_lr
exist_lambda = args.exist_lambda
kl_lambda = args.kl_lambda
kdec = args.kdec
encode_parts = args.encode_parts
predict_parts = args.predict_parts
penal_parts = args.penal_parts
predict_part_exist = args.predict_part_exist
variational = args.variational

hdim = args.hdim
n_train_epochs = args.n_epochs

encoder = PointNet(conv_dims=[32, 64, 64, 128, hdim * (2 if variational else 1)])
decoder = MLP(hdim, [256, 384, out_n*3], b_norm=bnorm, dropout_rate=drop)
part_classifier = PointNet(hdim+3, [64, n_parts], max_pool=False)
part_exister = MLP(hdim, [64, n_parts])  # MLP(hdim, [64, 64, n_parts])


# In[5]:


part_aware_model = True # or True

model = FancyPartAwarePointcloudAutoencoder(
    encoder, decoder, part_classifier, part_exister,
    part_lambda=part_lambda, device=device, class_decay=cdec, decode_alpha=alpha,
    variational=variational, kl_lambda=kl_lambda, exist_lambda=exist_lambda, noise=noise,
    kl_decay=kdec, encode_parts=encode_parts, predict_parts=predict_parts,
    predict_part_exist=predict_part_exist, penal_parts=penal_parts).to(device) # Students Work here

print(model)

model_tag = (f'exp' + 
             f'_outn{out_n}' + 
             f'_hdim{hdim}' + 
             f'_drop{drop}' + 
             f'{"_bnorm" if bnorm else ""}' + 
             f'_kl{kl_lambda:.0e}' + 
             f'_noisy{noise}' + 
             f'_cdec{cdec}' + 
             f'_alpha{alpha}' +
             f'_kdec{kdec}' + 
             f'{f"_predpts{part_lambda}" if predict_parts else ""}' +
             f'{f"_penpts{penal_parts}" if penal_parts > 0 else ""}' + 
             f'{f"_predex{exist_lambda}" if predict_part_exist else ""}' + 
             f'{"_encpts" if encode_parts else ""}' +
             f'{"_var" if variational else ""}' + 
             '_')


# In[6]:


optimizer = optim.Adam(model.parameters(), lr=init_lr)  # Students uncomment once you have defined your model


# In[7]:


min_val_loss = np.Inf
out_file = osp.join(top_out_dir, model_tag + 'best_model.pth')
start_epoch = 1


# In[8]:


## Train for multiple epochs your model.
# Students: the below for-loops are optional, feel free to structure your training 
# differently.

writer = SummaryWriter(log_dir=osp.join(top_log_dir, model_tag + time.strftime("%Y%m%d-%H%M%S")))
enc_dist = EncodingDistance('./cs233_gtda_hw4/data/golden_dists.npz')

for epoch in tqdm(range(start_epoch, start_epoch + n_train_epochs)):
    for phase in ['train', 'val', 'test']:
        # Students Work Here.
        if phase == 'train':
            epoch_losses, epoch_recon_losses, epoch_xentr_losses, epoch_bce_losses, epoch_kl_losses = model.train_for_one_epoch(loaders[phase], optimizer, device)
                
            if model.variational:
                writer.add_scalar('Loss/kl/train', epoch_kl_losses, epoch)
            if model.predict_parts:
                writer.add_scalar('Loss/xentr/train', epoch_xentr_losses, epoch)
            if model.predict_part_exist:
                writer.add_scalar('Loss/bce/train', epoch_bce_losses, epoch)
                
            writer.add_scalar('Loss/total/train', epoch_losses, epoch)
            writer.add_scalar('Loss/recon/train', epoch_recon_losses, epoch)
        else:
            _, _, val_losses, val_recon_losses, val_xentr_losses, val_bce_losses, val_kl_losses = model.reconstruct(loaders[phase], device)
                
            if model.variational:
                writer.add_scalar(f'Loss/kl/{phase}', val_kl_losses, epoch)
            if model.predict_parts:
                writer.add_scalar(f'Loss/xentr/{phase}', val_xentr_losses, epoch)
            if model.predict_part_exist:
                writer.add_scalar(f'Loss/bce/{phase}', val_bce_losses, epoch)
            
            writer.add_scalar(f'Loss/total/{phase}', val_losses, epoch)
            writer.add_scalar(f'Loss/recon/{phase}', val_recon_losses, epoch)
            
            # CHECK ENCODING DISTANCES EVERY 50 ITERS
            if epoch % 50 == 0:
                test_names = []
                latent_codes = []
                for load in loaders[phase]:
                    pointclouds = load['point_cloud']
                    test_names += load['model_name']
                    latent_codes += [l.cpu().numpy() for l in model.embed(pointclouds.to(device))]
                encoding_distance = enc_dist.calculate(latent_codes, test_names)['enc_dist']
                writer.add_scalar(f'Loss/enc/{phase}', encoding_distance, epoch)
                print(f'Epoch {epoch} distance: {encoding_distance}')

        # Save model if validation loss improved.
        if phase == 'val' and val_recon_losses < min_val_loss:
            min_val_loss = val_recon_losses
           
            # If you save the model like this, you can use the code below to load it. 
            save_state_dicts(out_file, epoch=epoch, model=model) 


# In[9]:


# Load model with best per-validation loss (uncomment when ready)
best_epoch = load_state_dicts(out_file, model=model, map_location=torch.device(device))
print('per-validation optimal epoch', best_epoch)
print('per-validation optimal recon loss', f'{min_val_loss.item():.3e}')

# Last, save the latent codes of the test data and go to the 
# measuring_part_awareness and tsne_plot_with_latent_codes code.

model.eval()
latent_codes = []
test_names = []
for load in loaders['test']:
    pointclouds = load['point_cloud']
    test_names += load['model_name']
    latent_codes += [l.cpu().numpy() for l in model.embed(pointclouds.to(device))]
    
latent_codes = np.array(latent_codes)

# Students TODO: Extract the latent codes and save them, so you can analyze them later.
np.savez(osp.join(top_out_dir, model_tag +'_latent_codes'), 
         latent_codes=latent_codes, 
         test_names=test_names)
