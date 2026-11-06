import argparse
import copy
import numpy as np
from run_train import run_training
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Menopause Classification Training")
    parser.add_argument('--model_name', type=str, default='resnet18', 
                        choices=['resnet18', 'resnet50', 'vit_b_16', 'vit_b_32', 'densenet121', 'convnext_tiny', 'efficientnet_b0', 'mobilenet_v3_large'],
                        help='Name of the model to train.')
    parser.add_argument('--save_name',type=str)
    parser.add_argument("--save_root",type=str,default ="./models/1_baseline/checkpoints")
    parser.add_argument('--removal', nargs='+', default=["./data/list/Removal.csv"], help="csv list of images to exclude from training")    
    parser.add_argument('--data_root',type=str,default="./data/anon_data_flatten_cropped_aligned_masking")
    parser.add_argument('--img_size', type=int, default=224, help='Input image size (default: 224)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs (default: 30)')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate (default: 2e-5)')
    parser.add_argument('--weight_decay', type=float, default=5e-3, help='Weight decay for optimizer (default: 5e-3)')
    parser.add_argument('--use_focal_loss', action='store_true', help='Use Focal Loss instead of standard CrossEntropyLoss')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma parameter for Focal loss (default: 2.0. Higher = more focus on hard examples)')
    parser.add_argument('--resize_mode',type=str,default='padded',choices=['random','padded','zeropad','reflect'])
    parser.add_argument('--num_runs', type=int, default=5, help="number of runs")
    args = parser.parse_args()

    SEEDS = [42, 777, 2024, 1004, 8282][:args.num_runs]

    results=[]

    for idx, seed in enumerate(SEEDS):
        print(f"\n{'='*60}")
        print(f"Starting run {idx+1}/{len(SEEDS)} (seed: {seed})")
        print(f"{'='*60}")
        
        current_args = copy.deepcopy(args)
        current_args.seed = seed
        current_args.save_name = f"{args.save_name}_seed{seed}"
        
        test_auc, weight_path = run_training(current_args)
        
        results.append({
            'seed': seed,
            'auc': test_auc,
            'path': weight_path
        })

    auc_list = [r['auc'] for r in results]
    mean_auc, std_auc = np.mean(auc_list), np.std(auc_list)
    
    representative_run = min(results, key=lambda x: abs(x['auc'] - mean_auc))
