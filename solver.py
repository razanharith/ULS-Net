import os
import numpy as np
import time
from datetime import datetime
from PIL import Image

import torch
import re
from ptflops import get_model_complexity_info
# from fvcore.nn import FlopCountAnalysis # This import is present but FlopCountAnalysis is not used, can be removed if not used elsewhere.

# from functions import cross_entropy_loss_RCF # This import is present but cross_entropy_loss_RCF is not used, can be removed.
import torchvision
from UNet import U_Net

# from UNet_VGG import UNet_VGG # Commented out
from torch import optim
# from torch.autograd import Variable # This import is present but Variable is not used, can be removed.
import torch.nn.functional as F
from evaluation import *
# from swin_transformer import SwinTransformer # Commented out
# from network import U_Net # Commented out
import cv2
import segmentation_models_pytorch as smp
import csv
from misc import *
import os
import argparse
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from tensorboardX import SummaryWriter
from ULS_MSA.ULS_MSA_v2 import ULS_MSA_v2
from ULS_MSA.ablation_models import ULS_MSA_NoEEM_NoSEM, ULS_MSA_SEMOnly, ULS_MSA_EEMOnly

writer = SummaryWriter('mylogdir')


class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):

        # Data loader
        self.mode = config.mode
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Hyper-parameters
        self.lr = config.lr
        self.optimizer_type = config.optimizer
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.num_epochs_decay = config.num_epochs_decay

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.SR_path = config.SR_path
        self.model_type = config.model_type  # Get model type from config
        self.dataset = config.dataset
        self.loss = config.loss_type

        # Report file
        self.report_file = config.report_file

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch

        self.augmentation_prob = config.augmentation_prob
        
   
        
        # Enhanced device detection for MPS (Apple Silicon) and CUDA
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("🍎 Using Apple Metal Performance Shaders (MPS)")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("🚀 Using NVIDIA CUDA")
        else:
            self.device = torch.device('cpu')
            print("⚠️  Using CPU (no GPU acceleration available)")
            
        print(f"Device: {self.device}")
        
        self.criterion1 = torch.nn.BCEWithLogitsLoss().to(self.device)
        self.criterion2 = mIoULoss(threshold=config.loss_threshold).to(self.device)
        self.criterion3 = DiceLoss(threshold=config.loss_threshold).to(self.device)

        # Model-specific parameters for VMUNet (with defaults)
        self.depths = getattr(config, 'depths', [2, 2, 9, 2])
        self.depths_decoder = getattr(config, 'depths_decoder', [2, 9, 2, 2])
        self.dims = getattr(config, 'dims', [48, 96, 192, 384])  # Compact dims for ~17M params
        self.drop_path_rate = getattr(config, 'drop_path_rate', 0.2)
        self.load_ckpt_path = getattr(config, 'load_ckpt_path', None)

        

       


    def build_model(self):
        """Build generator and discriminator."""
        print("initialize training...")
        
        # Model selection for ablation study
        if self.model_type == 'ULS_MSA_v2':
            self.unet = ULS_MSA_v2(img_ch=self.img_ch, output_ch=self.output_ch)
            print("Using full ULS_MSA_v2 model (with EEM and SEM)")
            
        elif self.model_type == 'ULS_MSA_NoEEM_NoSEM':
            self.unet = ULS_MSA_NoEEM_NoSEM(img_ch=self.img_ch, output_ch=self.output_ch)
            print("Using basic model without EEM and SEM")
            
        elif self.model_type == 'ULS_MSA_SEMOnly':
            self.unet = ULS_MSA_SEMOnly(img_ch=self.img_ch, output_ch=self.output_ch)
            print("Using model with only SEM attention")
            
        elif self.model_type == 'ULS_MSA_EEMOnly':
            self.unet = ULS_MSA_EEMOnly(img_ch=self.img_ch, output_ch=self.output_ch)
            print("Using model with only EEM for edge enhancement")
            
        elif self.model_type == 'LM_Net':
            self.unet = LM_Net(img_ch=self.img_ch, output_ch=self.output_ch)
            
        elif self.model_type == 'DINOv3_Segmentor':
            print("Loading DINOv3 Segmentor as backbone...")
            self.unet = load_dinov3_segmentor(device=str(self.device))
            print("DINOv3 Segmentor loaded.")

        else:
            self.unet = U_Net(self.img_ch, self.output_ch)


        if self.optimizer_type == 'Adam':
            self.optimizer = optim.Adam(self.unet.parameters(),self.lr, [self.beta1, self.beta2], weight_decay=1e-4)
        else:
            self.optimizer = optim.SGD(self.unet.parameters(), lr=self.lr, momentum=self.beta1, weight_decay=2e-4)

        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.8, patience=self.num_epochs_decay)

        self.unet.to(self.device)

        # Load pretrained weights for models that have load_from method
        if hasattr(self.unet, 'load_from') and self.load_ckpt_path is not None:
            print(f"Loading pretrained weights from {self.load_ckpt_path}")
            self.unet.load_from()

        #self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        trainable_params = 0
        for p in model.parameters():
            num_params += p.numel()
            if p.requires_grad:
                trainable_params += p.numel()
        
        print(f"\n{'='*50}")
        print(f"Model: {name}")
        print(f"{'='*50}")
        print(str(model))
        print(f"\nTotal parameters: {num_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"{'='*50}\n")
        
        self.report.write(f"\n{'-'*50}\n")
        self.report.write(f"Model: {name}\n")
        self.report.write(str(model))
        print(name)
        self.report.write('\n'+str(name))
        print("The number of parameters: {}".format(num_params))
        self.report.write("\n The number of parameters: {}".format(num_params))

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()


    def train(self,loss):
        factor = 0.8
        t = time.time()
        self.loss = loss
        isExist = os.path.exists(self.result_path + self.model_type+ '_' + loss)
        if not isExist:
            os.makedirs(self.result_path + self.model_type + '_' + loss)
        self.result_path_loss = os.path.join(self.result_path, self.model_type + '_' + loss) + '/' # Corrected path concatenation
        self.report = open(
            self.result_path_loss+ self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss + '.txt',
            'a+')
        self.report.write('\n' + str(datetime.now()))

        self.f1 = open(os.path.join(self.result_path_loss,
                                    self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss + '_train.csv'),
                       'a', encoding='utf-8', newline='')
        self.f2 = open(os.path.join(self.result_path_loss,
                                    self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss + '_val.csv'),
                       'a', encoding='utf-8', newline='')
        self.model_save_path = os.path.join(self.model_path,
                                            self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss + '.pkl')
        self.model_save_path1 = os.path.join(self.model_path,
                                            self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss)

        self.build_model()
        wr1 = csv.writer(self.f1)
        wr1.writerow(
            ['Epoch', 'Acc', 'RC', 'SP', 'PC', 'F1', 'IoU', 'mIoU', 'DC',
             'LR', 'loss'])
        wr2 = csv.writer(self.f2)
        wr2.writerow(
            ['Epoch', 'Acc', 'RC', 'SP', 'PC', 'F1', 'IoU', 'mIoU', 'DC',
             'LR', 'loss'])

        # U-Net Train
        if os.path.isfile(self.model_save_path):
            try:
                # Try loading with weights_only=False for backward compatibility with older PyTorch models
                self.unet = torch.load(self.model_save_path, weights_only=False)
                print('%s is Successfully Loaded from %s'%(self.model_type,self.model_save_path))
                self.report.write('\n %s is Successfully Loaded from %s'%(self.model_type,self.model_save_path))
            except Exception as e:
                print(f"Warning: Could not load existing model from {self.model_save_path}")
                print(f"Error: {e}")
                print("Starting training from scratch...")
                self.report.write(f'\n Warning: Could not load existing model from {self.model_save_path}')
                self.report.write(f'\n Error: {e}')
                self.report.write('\n Starting training from scratch...')
                # Continue with fresh training
                best_unet_score = 0.
                results = [["Loss",[],[]],["Acc",[],[]],["RC",[],[]],["SP",[],[]],["PC",[],[]],["F1",[],[]],["IoU",[],[]],["mIoU",[],[]],["DC",[],[]]]

                for epoch in range(self.num_epochs):
                    self.unet.train(True)
                    train_loss = 0.

                    acc = 0.
                    RC = 0.
                    SP = 0.
                    PC = 0.
                    F1 = 0.
                    IoU = 0
                    mIoU = 0.
                    DC = 0.
                    length = 0
                    buff = []

                    for i, (image, GT, name) in enumerate(self.train_loader):
                        # print('image')
                        # print(i)
                        # SR : Segmentation Result
                        # GT : Ground Truth
                        image = image.to(self.device)
                        GT = GT.to(self.device)
    # ----------------------------------UNet--------------------------------------------------------------

                        SR = self.unet(image)
                        
                        # Handle different model output formats
                        if self.model_type in ['Mobilenetv1', 'Mobilenetv2', 'Mobilenetv3', 'Mobilenetv4','ULS_MSA_v2','ULS_MSA_NoEEM_NoSEM','ULS_MSA_SEMOnly','ULS_MSA_EEMOnly']:
                            # MobileNet models return output directly
                            SR = SR.view(-1)
                        elif self.model_type in ['VMUNet', 'VMUNetV2', 'LightMUNet','ULS_MSA_v2']:
                            # These models return single tensor output directly
                            SR = SR.view(-1)
                        else:
                            # Other models return output as tuple/list
                            SR = SR[0]
                            SR = SR.view(-1)
                    
                        GT = GT.view(-1)

                        loss1 = self.criterion1(SR, GT)
                        loss2 = self.criterion2(SR, GT)
                        loss3 = self.criterion3(SR,GT)

                        #total_loss = loss1 + loss2 + loss3
                        total_loss = loss1 + (factor*(loss2+loss3))

                        self.reset_grad()
                        total_loss.backward()
                        self.optimizer.step()

                        SR = SR.detach()
                        GT = GT.detach()

                        train_loss += total_loss.detach().item()
                        acc += get_accuracy(SR,GT)
                        RC += get_Recall(SR,GT)
                        SP += get_specificity(SR,GT)
                        PC += get_Precision(SR,GT)
                        F1 += get_F1(SR,GT)
                        buff = get_mIoU(SR,GT)
                        IoU += buff[0]
                        mIoU += buff[1]
                        DC += get_DC(SR,GT)
                        length += 1

                    train_loss = train_loss/length
                    acc = acc/length
                    RC = RC/length
                    SP = SP/length
                    PC = PC/length
                    F1 = F1/length
                    IoU = IoU/length
                    mIoU = mIoU/length
                    DC = DC/length

                    results[0][1].append((train_loss))
                    results[1][1].append((acc*100))
                    results[2][1].append((RC*100))
                    results[3][1].append((SP*100))
                    results[4][1].append((PC*100))
                    results[5][1].append((F1*100))
                    results[6][1].append((IoU*100))
                    results[7][1].append((mIoU*100))
                    results[8][1].append((DC*100))

                    print('\nEpoch [%d/%d] \nTrain Loss: %.4f \n[Training] Acc: %.4f, RC: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, DC: %.4f' % (
                        epoch+1,self.num_epochs,train_loss,acc,RC,SP,PC,F1,IoU,mIoU,DC))
                    self.report.write('\nEpoch [%d/%d] \nTrain Loss: %.4f \n[Training] Acc: %.4f, RC: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, DC: %.4f' % (
                        epoch+1,self.num_epochs,train_loss,acc,RC,SP,PC,F1,IoU,mIoU,DC))
                    wr1.writerow(
                        [epoch + 1, acc, RC, SP, PC, F1, IoU, mIoU, DC, self.lr, train_loss])
                    writer.add_scalar("Loss/train", train_loss, epoch+1)
                    writer.add_scalar("Precision/train", PC, epoch + 1)
                    writer.add_scalar("Recall/train", RC, epoch + 1)
                    writer.add_scalar("F1 Score/train", F1, epoch + 1)
                    writer.add_scalar("mIoU/train", mIoU, epoch + 1)

                    # Memory cleanup for both CUDA and MPS
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()

    #===================================== Validation ====================================#
                    self.unet.train(False)
                    self.unet.eval()
                    valid_loss = 0.

                    acc = 0.
                    RC = 0.
                    SP = 0.
                    PC = 0.
                    F1 = 0.
                    IoU = 0
                    mIoU = 0.
                    DC = 0.
                    length=0
                    buff = []

                    for i, (image, GT, name) in enumerate(self.valid_loader):
                        
                        # SR : Segmentation Result
                        # GT : Ground Truth
                        image = image.to(self.device)
                        GT = GT.to(self.device)
                        GT_original = GT  # Keep original for image saving
                        GT_f = GT

    #-------------------------------------UNet-------------------------------------------------------
                        SR = self.unet(image)
                        
                        # Handle different model output formats
                        if self.model_type in ['Mobilenetv1', 'Mobilenetv2', 'Mobilenetv3', 'Mobilenetv4', 'VMUNet', 'VMUNetV2', 'LightMUNet','ULS_MSA_v2','ULS_MSA_NoEEM_NoSEM','ULS_MSA_SEMOnly','ULS_MSA_EEMOnly']:
                            # MobileNet, VMUNet, and LightMUNet models return output directly
                            SR_original = SR  # Keep original for image saving
                            SR_f = SR.view(-1)
                        else:
                            # Other models return output as tuple/list
                            SR_original = SR[0]  # Keep original for image saving
                            SR = SR[0]
                            SR_f = SR.view(-1)
                      
                        GT_f = GT.view(-1)
                        loss_val_1 = self.criterion1(SR_f, GT_f)
                        loss_val_2 = self.criterion2(SR_f, GT_f)
                        loss_val_3 = self.criterion3(SR_f,GT_f)

                        #total_loss = loss_val_1 
                        total_loss = loss_val_1 + (factor*(loss_val_2+loss_val_3))

                        # Apply sigmoid to convert logits to probabilities for metric computation
                        SR_f_prob = torch.sigmoid(SR_f.detach())
                        GT_f = GT_f.detach()

                        valid_loss += total_loss.detach().item()
                        acc += get_accuracy(SR_f_prob,GT_f)
                        RC += get_Recall(SR_f_prob,GT_f)
                        SP += get_specificity(SR_f_prob,GT_f)
                        PC += get_Precision(SR_f_prob,GT_f)
                        F1 += get_F1(SR_f_prob,GT_f)
                        buff = get_mIoU(SR_f_prob,GT_f)
                        IoU += buff[0]
                        mIoU += buff[1]
                        DC += get_DC(SR_f_prob,GT_f)
                        length += 1

                    valid_loss = valid_loss/length
                    acc = acc/length
                    RC = RC/length
                    SP = SP/length
                    PC = PC/length
                    F1 = F1/length
                    IoU = IoU/length
                    mIoU = mIoU/length
                    DC = DC/length
                    unet_score = mIoU

                    results[0][2].append((valid_loss))
                    results[1][2].append((acc*100))
                    results[2][2].append((RC*100))
                    results[3][2].append((SP*100))
                    results[4][2].append((PC*100))
                    results[5][2].append((F1*100))
                    results[6][2].append((IoU*100))
                    results[7][2].append((mIoU*100))
                    results[8][2].append((DC*100))

                    print('\nVal Loss: %.4f \n[Validation] Acc: %.4f, RC: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, DC: %.4f'%(
                        valid_loss,acc,RC,SP,PC,F1,IoU,mIoU,DC))
                    self.report.write('\nVal Loss: %.4f \n[Validation] Acc: %.4f, RC: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, DC: %.4f'%(
                        valid_loss,acc,RC,SP,PC,F1,IoU,mIoU,DC))

                    wr2.writerow([epoch+1 ,acc,RC,SP,PC,F1,IoU,mIoU,DC,self.lr,valid_loss])
                    writer.add_scalar("Loss/val", valid_loss, epoch + 1)
                    writer.add_scalar("Precision/val", PC, epoch + 1)
                    writer.add_scalar("Recall/val", RC, epoch + 1)
                    writer.add_scalar("F1 Score/val", F1, epoch + 1)
                    writer.add_scalar("mIoU/val", mIoU, epoch + 1)


                    self.lr_scheduler.step(valid_loss)

                    if unet_score > best_unet_score:
                        best_unet_score = unet_score
                        print('\nBest %s model score : %.4f'%(self.model_type,best_unet_score))
                        self.report.write('\nBest %s model score : %.4f'%(self.model_type,best_unet_score))
                        torch.save(self.unet,self.model_save_path)
                    epoch_custom = epoch + 1
                    if epoch_custom % 10 ==0:
                        torch.save(self.unet, self.model_save_path1+'_'+str(epoch_custom)+'.pkl')


                    if unet_score > 0.9:
                        torchvision.utils.save_image(image.data.cpu(),os.path.join(
                            self.result_path_loss,self.report_file+'_%s_valid_%d_image.png'%(self.model_type,epoch+1)))
                        torchvision.utils.save_image(torch.sigmoid(SR_original).data.cpu(),os.path.join(
                            self.result_path_loss,self.report_file+'_%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
                        torchvision.utils.save_image(GT_original.data.cpu(),os.path.join(
                            self.result_path_loss,self.report_file+'_%s_valid_%d_GT.png'%(self.model_type,epoch+1)))

                    # Memory cleanup for both CUDA and MPS
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()

                displayfigures(results, self.result_path_loss, self.report_file, self.dataset, self.model_type)
        else:
            best_unet_score = 0.
            results = [["Loss",[],[]],["Acc",[],[]],["RC",[],[]],["SP",[],[]],["PC",[],[]],["F1",[],[]],["IoU",[],[]],["mIoU",[],[]],["DC",[],[]]]

            for epoch in range(self.num_epochs):
                self.unet.train(True)
                train_loss = 0.

                acc = 0.
                RC = 0.
                SP = 0.
                PC = 0.
                F1 = 0.
                IoU = 0
                mIoU = 0.
                DC = 0.
                length = 0
                buff = []

                for i, (image, GT, name) in enumerate(self.train_loader):
                    # print('image')
                    # print(i)
                    # SR : Segmentation Result
                    # GT : Ground Truth
                    image = image.to(self.device)
                    GT = GT.to(self.device)
# ----------------------------------UNet--------------------------------------------------------------

                    SR = self.unet(image)
                    
                    # Handle different model output formats
                    if self.model_type == 'DINOv3_Segmentor':
                        # Keep spatial dimensions for DINOv3
                        B = SR.shape[0]
                        SR_flat = SR.view(B, -1)
                        GT_flat = GT.view(B, -1)
                    elif self.model_type in ['Mobilenetv1', 'Mobilenetv2', 'Mobilenetv3', 'Mobilenetv4', 'VMUNet', 'VMUNetV2','LightMUNet','ULS_MSA_v2','ULS_MSA_NoEEM_NoSEM','ULS_MSA_SEMOnly','ULS_MSA_EEMOnly']:
                        # MobileNet and VMUNet models return output directly
                        SR_flat = SR.view(-1)
                        GT_flat = GT.view(-1)
                    else:
                        # Other models return output as tuple/list
                        SR = SR[0]
                        SR_flat = SR.view(-1)
                        GT_flat = GT.view(-1)
                    
                    # Use flattened versions for loss computation
                    SR_for_loss = SR_flat
                    GT_for_loss = GT_flat

                    loss1 = self.criterion1(SR_for_loss, GT_for_loss)
                    loss2 = self.criterion2(SR_for_loss, GT_for_loss)
                    loss3 = self.criterion3(SR_for_loss, GT_for_loss)

                    total_loss = loss1 + (factor*(loss2+loss3))

                    self.reset_grad()
                    total_loss.backward()
                    self.optimizer.step()

                    # Apply sigmoid to convert logits to probabilities for metric computation
                    with torch.no_grad():
                        if self.model_type == 'DINOv3_Segmentor':
                            SR_prob = torch.sigmoid(SR.detach())
                            SR_prob_flat = SR_prob.view(B, -1)
                        else:
                            SR_prob_flat = torch.sigmoid(SR_for_loss.detach())
                        
                        GT_metric = GT_for_loss.detach()

                    train_loss += total_loss.detach().item()
                    acc += get_accuracy(SR_prob_flat, GT_metric)
                    RC += get_Recall(SR_prob_flat, GT_metric)
                    SP += get_specificity(SR_prob_flat, GT_metric)
                    PC += get_Precision(SR_prob_flat, GT_metric)
                    F1 += get_F1(SR_prob_flat, GT_metric)
                    buff = get_mIoU(SR_prob_flat, GT_metric)
                    IoU += buff[0]
                    mIoU += buff[1]
                    DC += get_DC(SR_prob_flat, GT_metric)
                    length += 1

                train_loss = train_loss/length
                acc = acc/length
                RC = RC/length
                SP = SP/length
                PC = PC/length
                F1 = F1/length
                IoU = IoU/length
                mIoU = mIoU/length
                DC = DC/length

                results[0][1].append((train_loss))
                results[1][1].append((acc*100))
                results[2][1].append((RC*100))
                results[3][1].append((SP*100))
                results[4][1].append((PC*100))
                results[5][1].append((F1*100))
                results[6][1].append((IoU*100))
                results[7][1].append((mIoU*100))
                results[8][1].append((DC*100))

                print('\nEpoch [%d/%d] \nTrain Loss: %.4f \n[Training] Acc: %.4f, RC: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, DC: %.4f' % (
                    epoch+1,self.num_epochs,train_loss,acc,RC,SP,PC,F1,IoU,mIoU,DC))
                self.report.write('\nEpoch [%d/%d] \nTrain Loss: %.4f \n[Training] Acc: %.4f, RC: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, DC: %.4f' % (
                    epoch+1,self.num_epochs,train_loss,acc,RC,SP,PC,F1,IoU,mIoU,DC))
                wr1.writerow(
                    [epoch + 1, acc, RC, SP, PC, F1, IoU, mIoU, DC, self.lr, train_loss])
                writer.add_scalar("Loss/train", train_loss, epoch+1)
                writer.add_scalar("Precision/train", PC, epoch + 1)
                writer.add_scalar("Recall/train", RC, epoch + 1)
                writer.add_scalar("F1 Score/train", F1, epoch + 1)
                writer.add_scalar("mIoU/train", mIoU, epoch + 1)

                # Memory cleanup for both CUDA and MPS
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()

#===================================== Validation ====================================#
                self.unet.train(False)
                self.unet.eval()
                valid_loss = 0.

                acc = 0.
                RC = 0.
                SP = 0.
                PC = 0.
                F1 = 0.
                IoU = 0
                mIoU = 0.
                DC = 0.
                length=0
                buff = []

                for i, (image, GT, name) in enumerate(self.valid_loader):
                    
                    # SR : Segmentation Result
                    # GT : Ground Truth
                    image = image.to(self.device)
                    GT = GT.to(self.device)
                    GT_original = GT  # Keep original for image saving
                    GT_f = GT

#-------------------------------------UNet-------------------------------------------------------
                    SR = self.unet(image)
                    
                    # Handle different model output formats
                    if self.model_type == 'DINOv3_Segmentor':
                        # Keep spatial dimensions for DINOv3
                        SR_original = SR  # Keep original for image saving
                        B = SR.shape[0]
                        SR_f = SR.view(B, -1)
                        GT_f = GT.view(B, -1)
                    elif self.model_type in ['Mobilenetv1', 'Mobilenetv2', 'Mobilenetv3', 'Mobilenetv4', 'VMUNet', 'VMUNetV2','LightMUNet','ULS_MSA_v2','ULS_MSA_NoEEM_NoSEM','ULS_MSA_SEMOnly','ULS_MSA_EEMOnly']:
                        # MobileNet and VMUNet models return output directly
                        SR_original = SR  # Keep original for image saving
                        SR_f = SR.view(-1)
                        GT_f = GT.view(-1)
                    else:
                        # Other models return output as tuple/list
                        SR_original = SR[0]  # Keep original for image saving
                        SR = SR[0]
                        SR_f = SR.view(-1)
                        GT_f = GT.view(-1)
                   
                    loss_val_1 = self.criterion1(SR_f, GT_f)
                    loss_val_2 = self.criterion2(SR_f, GT_f)
                    loss_val_3 = self.criterion3(SR_f, GT_f)

                    total_loss = loss_val_1 + (factor*(loss_val_2+loss_val_3))

                    # Apply sigmoid to convert logits to probabilities for metric computation
                    with torch.no_grad():
                        if self.model_type == 'DINOv3_Segmentor':
                            SR_prob = torch.sigmoid(SR.detach())
                            SR_prob_flat = SR_prob.view(B, -1)
                        else:
                            SR_prob_flat = torch.sigmoid(SR_f.detach())
                        GT_metric = GT_f.detach()

                    valid_loss += total_loss.detach().item()
                    acc += get_accuracy(SR_prob_flat, GT_metric)
                    RC += get_Recall(SR_prob_flat, GT_metric)
                    SP += get_specificity(SR_prob_flat, GT_metric)
                    PC += get_Precision(SR_prob_flat, GT_metric)
                    F1 += get_F1(SR_prob_flat, GT_metric)
                    buff = get_mIoU(SR_prob_flat, GT_metric)
                    IoU += buff[0]
                    mIoU += buff[1]
                    DC += get_DC(SR_prob_flat, GT_metric)
                    length += 1

                valid_loss = valid_loss/length
                acc = acc/length
                RC = RC/length
                SP = SP/length
                PC = PC/length
                F1 = F1/length
                IoU = IoU/length
                mIoU = mIoU/length
                DC = DC/length
                unet_score = mIoU

                results[0][2].append((valid_loss))
                results[1][2].append((acc*100))
                results[2][2].append((RC*100))
                results[3][2].append((SP*100))
                results[4][2].append((PC*100))
                results[5][2].append((F1*100))
                results[6][2].append((IoU*100))
                results[7][2].append((mIoU*100))
                results[8][2].append((DC*100))

                print('\nVal Loss: %.4f \n[Validation] Acc: %.4f, RC: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, DC: %.4f'%(
                    valid_loss,acc,RC,SP,PC,F1,IoU,mIoU,DC))
                self.report.write('\nVal Loss: %.4f \n[Validation] Acc: %.4f, RC: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, DC: %.4f'%(
                    valid_loss,acc,RC,SP,PC,F1,IoU,mIoU,DC))

                wr2.writerow([epoch+1 ,acc,RC,SP,PC,F1,IoU,mIoU,DC,self.lr,valid_loss])
                writer.add_scalar("Loss/val", valid_loss, epoch + 1)
                writer.add_scalar("Precision/val", PC, epoch + 1)
                writer.add_scalar("Recall/val", RC, epoch + 1)
                writer.add_scalar("F1 Score/val", F1, epoch + 1)
                writer.add_scalar("mIoU/val", mIoU, epoch + 1)


                self.lr_scheduler.step(valid_loss)

                if unet_score > best_unet_score:
                    best_unet_score = unet_score
                    print('\nBest %s model score : %.4f'%(self.model_type,best_unet_score))
                    self.report.write('\nBest %s model score : %.4f'%(self.model_type,best_unet_score))
                    torch.save(self.unet,self.model_save_path)
                epoch_custom = epoch + 1
                if epoch_custom % 10 ==0:
                    torch.save(self.unet, self.model_save_path1+'_'+str(epoch_custom)+'.pkl')


                if unet_score > 0.9:
                    torchvision.utils.save_image(image.data.cpu(),os.path.join(
                        self.result_path_loss,self.report_file+'_%s_valid_%d_image.png'%(self.model_type,epoch+1)))
                
                    torchvision.utils.save_image(torch.sigmoid(SR_original).data.cpu(),os.path.join(
                        self.result_path_loss,self.report_file+'_%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
                    
                    torchvision.utils.save_image(GT_original.data.cpu(),os.path.join(
                        self.result_path_loss,self.report_file+'_%s_valid_%d_GT.png'%(self.model_type,epoch+1)))

              

                # Memory cleanup for both CUDA and MPS
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            displayfigures(results, self.result_path_loss, self.report_file, self.dataset, self.model_type)

        elapsed = time.time() - t
        print("\nElapsed time: %f seconds.\n\n" %elapsed)
        self.report.write("\nElapsed time: %f seconds.\n\n" %elapsed)
        self.report.close()
        self.f1.close()
        self.f2.close()
        writer.close()

    def get_gradCAM(self,image,SR, GT,size):
        total_loss = self.criterion1(SR, GT)
        total_loss.backward()
        gradients = self.unet.get_activation_gradients()
        pooled_gradients = torch.mean(gradients, dim=[0,2,3])
        activations = self.unet.get_activations(image).detach()
        for i in range(activations.shape[1]):
            activations[:,i,:,:] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim = 1).squeeze().cpu()
        heatmap = nn.ReLU()(heatmap)
        heatmap /= torch.max(heatmap)
        heatmap = np.uint8(255 * heatmap)
        image = image.squeeze(0)
        image = image.permute(1,2,0)
        image = image.cpu().numpy()
        image = np.uint8(image * 255)
        heatmap = cv2.resize(heatmap, (320, 320))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(heatmap, 0.5, image, 0.5, 0)
        return overlay, heatmap

    def test(self, loss, data, model): 
        if os.path.isfile(self.model_path + self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss + '.pkl'):
            try:
                # Try loading with weights_only=False for backward compatibility with older PyTorch models
                self.unet = torch.load(self.model_path + self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss + '.pkl', weights_only=False)
                print('%s is Successfully Loaded from %s' % (self.model_type, self.model_path))
            except Exception as e:
                print(f"Error loading model: {e}")
                print(f"Trained model NOT found or could not be loaded for {self.model_type} with loss {loss}, Please train a model first")
                return
        else:
            print(f"Trained model NOT found for {self.model_type} with loss {loss}, Please train a model first")
            return

        isExist = os.path.exists(self.SR_path + self.model_type + '_' + loss)
        if not isExist:
            os.makedirs(self.SR_path + self.model_type + '_' + loss)

        self.model_path_loss = self.SR_path + self.model_type + '_' + loss + '/'
        self.test_acc = open(self.model_path_loss + self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss + '_test.csv', 'a+')

        wr_test = csv.writer(self.test_acc)
        if os.path.getsize(self.model_path_loss + self.report_file + '_' + self.dataset + '_' + self.model_type + '_' + loss + '_test.csv') == 0:
            wr_test.writerow(['Accuracy', 'Recall', 'Precision', 'F1', 'mIoU', 'Dice', 'Params', 'FLOPs', 'Avg Inference Time'])


        self.unet.train(False)
        self.unet.eval()

        input_size_flops = (self.img_ch, 224, 224)
        try:
            macs, params = get_model_complexity_info(self.unet, input_size_flops, as_strings=True,
                                                         print_per_layer_stat=False, verbose=False)
            flops = eval(re.findall(r'([\d.]+)', macs)[0]) * 2
            flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]
        except Exception as e:
            print(f"Could not calculate FLOPs/Params for {self.model_type}: {e}")
            macs, params, flops, flops_unit = "N/A", "N/A", "N/A", ""

        print(f'Computational complexity: {macs}')
        print(f'Computational complexity: {flops} {flops_unit}Flops')
        print(f'Number of parameters: {params}')

        acc = 0.
        RC = 0.
        SP = 0.
        PC = 0.
        F1 = 0.
        IoU = 0
        mIoU = 0.
        DC = 0.
        total_elapsed = 0.
        length = 0

        with torch.no_grad():
            for i, (image, GT, name) in enumerate(self.test_loader):
                image = image.to(self.device)
                GT = GT.to(self.device)

                start_time = time.time()
                SR = self.unet(image)
                elapsed = time.time() - start_time
                total_elapsed += elapsed
                
                print(f"Raw output shape: {SR.shape}")  # Debug info
                
                # Handle different model output formats
                if self.model_type == 'DINOv3_Segmentor':
                    # Keep spatial dimensions for DINOv3
                    print(f"DINOv3 SR min/max before sigmoid: {SR.min().item():.4f}/{SR.max().item():.4f}")  # Debug info
                    SR_sigmoid = torch.sigmoid(SR)  # Apply sigmoid while keeping spatial dimensions
                    print(f"DINOv3 SR min/max after sigmoid: {SR_sigmoid.min().item():.4f}/{SR_sigmoid.max().item():.4f}")  # Debug info
                    SR_f = SR.view(-1)  # Flatten for metrics
                    SR_f_sigmoid = SR_sigmoid.view(-1)  # Flatten sigmoid output for metrics
                elif self.model_type in ['Mobilenetv1', 'Mobilenetv2', 'Mobilenetv3', 'Mobilenetv4', 'VMUNet', 'VMUNetV2','LightMUNet','ULS_MSA_v2']:
                    # MobileNet and VMUNet models return output directly
                    SR_f = SR.view(-1)
                    SR_f_sigmoid = torch.sigmoid(SR_f)
                    SR_sigmoid = torch.sigmoid(SR)
                else:
                    # Other models return output as tuple/list
                    SR = SR[0]  # Modify this based on your model's output structure
                    SR_f = SR.view(-1)
                    SR_f_sigmoid = torch.sigmoid(SR_f)
                    SR_sigmoid = torch.sigmoid(SR)
                GT_f = GT.view(-1)

                acc += get_accuracy(SR_f_sigmoid, GT_f)
                RC += get_Recall(SR_f_sigmoid, GT_f)
                SP += get_specificity(SR_f_sigmoid, GT_f)
                PC += get_Precision(SR_f_sigmoid, GT_f)
                F1 += get_F1(SR_f_sigmoid, GT_f)
                buff = get_mIoU(SR_f_sigmoid, GT_f)
                IoU += buff[0]
                mIoU += buff[1]
                DC += get_DC(SR_f_sigmoid, GT_f)
                length += 1

                threshold = 0.5
                
                if self.model_type == 'DINOv3_Segmentor':
                    # Process DINOv3 output specifically
                    print(f"DINOv3 SR_sigmoid shape before squeeze: {SR_sigmoid.shape}")  # Debug info
                    SR_processed = SR_sigmoid.squeeze(1) if SR_sigmoid.shape[1] == 1 else SR_sigmoid  # Remove channel dim only if it exists
                    print(f"DINOv3 SR_processed shape after squeeze: {SR_processed.shape}")  # Debug info
                    print(f"DINOv3 SR_processed min/max before threshold: {SR_processed.min().item():.4f}/{SR_processed.max().item():.4f}")  # Debug info
                    SR_thresholded = (SR_processed >= threshold).float()
                    
                    # Save both probability map and thresholded result
                    for j in range(SR_processed.shape[0]):
                        # Save probability map
                        prob_im = Image.fromarray((SR_processed[j].cpu().numpy() * 255).astype(np.uint8)).convert('L')
                        prob_im = prob_im.resize((256, 256), resample=Image.BILINEAR)
                        prob_im.save(self.model_path_loss + 'prob_' + name[j])
                        
                        # Save thresholded result
                        thresh_im = Image.fromarray((SR_thresholded[j].cpu().numpy() * 255).astype(np.uint8)).convert('L')
                        thresh_im = thresh_im.resize((256, 256), resample=Image.BILINEAR)
                        thresh_im.save(self.model_path_loss + name[j])
                        
                        # Debug: Save input image and ground truth
                        input_im = Image.fromarray((image[j, 0].cpu().numpy() * 255).astype(np.uint8)).convert('L')
                        input_im.save(self.model_path_loss + 'input_' + name[j])
                        
                        gt_im = Image.fromarray((GT[j, 0].cpu().numpy() * 255).astype(np.uint8)).convert('L')
                        gt_im.save(self.model_path_loss + 'gt_' + name[j])
                else:
                    # Original processing for other models
                    SR_processed = torch.sigmoid(SR).squeeze(1)
                    SR_processed[SR_processed < threshold] = 0
                    SR_processed[SR_processed >= threshold] = 1

                    for j in range(SR_processed.shape[0]):
                        im = Image.fromarray(SR_processed[j].cpu().numpy() * 255).convert('L')
                        imo = im.resize((256, 256), resample=Image.BILINEAR)
                        imo.save(self.model_path_loss + name[j])

        acc /= length
        RC /= length
        SP /= length
        PC /= length
        F1 /= length
        IoU /= length
        mIoU /= length
        DC /= length

        total_images_processed = length * self.test_loader.batch_size
        avg_inference_time = total_elapsed / total_images_processed if total_images_processed > 0 else 0

        wr_test.writerow([acc, RC, PC, F1, mIoU, DC, params, flops, avg_inference_time ])
        print('Results have been Saved')
        print(f'Average Inference Time per Image: {avg_inference_time:.6f} seconds')

        self.test_acc.close()