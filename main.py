import argparse
import os
from solver import Solver  # Updated to use solver_n which has EADSNet support
from data_loader import get_loader
from torch.backends import cudnn
import random

def main(config):
    cudnn.benchmark = True

    # Create model and result directories if they don't exist
    os.makedirs(config.model_path, exist_ok=True)
    os.makedirs(config.result_path, exist_ok=True)

    print(config)

    # Load data
    train_loader = get_loader(image_path=config.train_path,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='train',
                              augmentation_prob=config.augmentation_prob)

    valid_loader = get_loader(image_path=config.valid_path,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='valid',
                              augmentation_prob=0)

    test_loader = get_loader(image_path=config.test_path,
                             image_size=config.image_size,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers,
                             mode='test',
                             augmentation_prob=0)

    # Ablation Study - Testing models
    all_ablation_models = [
        'ULS_MSA_NoEEM_NoSEM', # Basic model without EEM and SEM
        'ULS_MSA_SEMOnly',     # Model with only SEM attention
        'ULS_MSA_EEMOnly'      # Model with only EEM for edge enhancement
    ]  # List of available models (ULS_MSA_v2 already trained)
    
    # Filter models based on command line argument
    if config.models:
        ablation_models = [model for model in config.models.split(',') if model in all_ablation_models]
        if not ablation_models:
            print(f"Warning: No valid models found. Available models: {all_ablation_models}")
            ablation_models = all_ablation_models  # Run all ablation variants
    else:
        ablation_models = all_ablation_models  # Run all ablation variants

    print(f"Running models: {ablation_models}")

    # Process each model
    for model_name in ablation_models:
        print(f"\n{'='*50}")
        print(f"Processing : {model_name}")
        print(f"{'='*50}")

        config.model_type = model_name  # Set the model type in config
        solver = Solver(config, train_loader, valid_loader, test_loader)

        if config.mode == 'train':
            # Training phase
            for loss in ['BCE_Dice_mIoU']:
                solver.train(loss=loss)

        elif config.mode == 'test':


            # Testing phase
            print(f"\n{'='*50}")
            print(f"Testing : {model_name}")
            print(f"{'='*50}")
            for loss in ['BCE_Dice_mIoU']:
                solver.test(loss=loss, data='PH2', model=model_name)

        else:
            print(f"Invalid mode: {config.mode}. Please use 'train' or 'test'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=224)     # Standard size for better results
    parser.add_argument('--num_workers', type=int, default=0)

    # Training hyper-parameters - Research-Proven for Medical Segmentation
    parser.add_argument('--lr', type=float, default=0.0001)       # Lower LR for stable training with combined loss
    parser.add_argument('--num_epochs', type=int, default=100)     # Standard for medical segmentation
    parser.add_argument('--num_epochs_decay', type=int, default=10) # T_0 for CosineAnnealingWarmRestarts
    parser.add_argument('--batch_size', type=int, default=2)       # Larger batch for stable gradients
    parser.add_argument('--loss_threshold', type=float, default=0.5)
    parser.add_argument('--loss_type', type=str, default='BCE_Dice_mIoU', help='[BCE,BCE_mIoU,BCE_Dice_mIoU]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='[Adam,SGD,AdamW]') # Standard Adam (research-proven)
    parser.add_argument('--beta1', type=float, default=0.9)        # Standard Adam momentum
    parser.add_argument('--beta2', type=float, default=0.999)      # Standard Adam momentum
    parser.add_argument('--weight_decay', type=float, default=0.0001)  # Increased weight decay for better regularization
    parser.add_argument('--augmentation_prob', type=float, default=0.9) # Strong augmentation for large datasets



    # Misc  
    parser.add_argument('--report_file', type=str, default='PH2')
    parser.add_argument('--mode', type=str, default='test', help='[train,test] - Note: Program will run both train and test')
    parser.add_argument('--dataset', type=str, default='PH2', help='[PH2,PH2,ISIC2018]')
    parser.add_argument('--use_enhanced_lmnet', action='store_true')
    parser.add_argument('--models', type=str, default=None, help='Comma-separated list of models to run')
    parser.add_argument('--save_images', action='store_true', help='Save predicted images during testing')

   
    parser.add_argument('--train_path', type=str, default='/Users/razan/Documents/Research/2.Technical/0.Datasets/PH2/train/')
    parser.add_argument('--valid_path', type=str, default='/Users/razan/Documents/Research/2.Technical/0.Datasets/PH2/valid/')
    parser.add_argument('--test_path', type=str, default='/Users/razan/Documents/Research/2.Technical/0.Datasets/PH2/test/')
    parser.add_argument('--model_path', type=str, default='/Users/razan/Documents/Research/2.Technical/5.ULS_MSA/code/Results/Ablation/models_PH2/')
    parser.add_argument('--result_path', type=str, default='/Users/razan/Documents/Research/2.Technical/5.ULS_MSA/code/Results/Ablation/results_PH2/')
    parser.add_argument('--SR_path', type=str, default='/Users/razan/Documents/Research/2.Technical/5.ULS_MSA/code/Results/Ablation/SR_isic017/')

    parser.add_argument('--cuda_idx', type=int, default=1)
    parser.add_argument('--model_type', type=str, default='ULS_MSA_v2', help='Model type for ablation study')

    config = parser.parse_args()
    
    # Automatically enable save_images when in test mode
    if config.mode == 'test' and not config.save_images:
        config.save_images = True
        print("🖼️  Auto-enabled save_images for test mode")
    
    main(config)