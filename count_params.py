#!/usr/bin/env python3
"""
Script to count parameters in different components of VMUNetV2
"""
import torch
from solver_n import Solver
import argparse

def count_parameters_by_component(model):
    """Count parameters in different parts of the model"""
    param_counts = {}
    
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        param_counts[name] = params
        print(f"{name}: {params:,} parameters ({params/1e6:.2f}M)")
    
    # Check VSSM internal components
    if hasattr(model, 'vmunet'):
        print("\nVSSM (vmamba backbone) internal components:")
        for name, module in model.vmunet.named_children():
            params = sum(p.numel() for p in module.parameters())
            print(f"  vmunet.{name}: {params:,} parameters ({params/1e6:.2f}M)")
    
    return param_counts

def main():
    # Create config with minimal dimensions - include all required attributes
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--depths', type=list, default=[2, 2, 9, 2])
    parser.add_argument('--depths_decoder', type=list, default=[2, 9, 2, 2])
    parser.add_argument('--dims', type=list, default=[24, 48, 96, 192])
    parser.add_argument('--drop_path_rate', type=float, default=0.2)
    parser.add_argument('--load_ckpt_path', type=str, default=None)
    # Add missing required attributes
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--model_path', type=str, default='./models/')
    parser.add_argument('--result_path', type=str, default='./results/')
    parser.add_argument('--train_path', type=str, default='./train/')
    parser.add_argument('--valid_path', type=str, default='./valid/')
    parser.add_argument('--test_path', type=str, default='./test/')
    parser.add_argument('--SR_path', type=str, default='./sr/')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_epochs_decay', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--loss_threshold', type=float, default=0.5)
    parser.add_argument('--augmentation_prob', type=float, default=0.4)
    config = parser.parse_args([])
    
    # Create solver and analyze parameters
    print("Creating VMUNetV2 with minimal configuration...")
    solver = Solver(config, 'VMUNetV2', None, None, None)
    
    total_params = sum(p.numel() for p in solver.unet.parameters())
    print(f'\nTotal VMUNetV2 Parameters: {total_params:,} ({total_params/1e6:.2f}M)')
    print(f'Target: 17.91M parameters')
    print(f'Difference: {(total_params - 17.91e6)/1e6:.2f}M parameters over target\n')
    
    print("Parameter breakdown by component:")
    print("="*50)
    count_parameters_by_component(solver.unet)
    
if __name__ == "__main__":
    main()
