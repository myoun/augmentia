import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import AugmentNet
from dataset import AugmentDataset

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f'[Augmentia Trainer] Using device: {device}')
    
    dataset = AugmentDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f'[Augmentia Trainer] Loaded {len(dataset)} training samples from {args.data}')
    
    model = AugmentNet(num_blocks=args.blocks, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    mse_loss_fn = nn.MSELoss()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_p_loss = 0.0
        total_v_loss = 0.0
        batches = 0
        
        for batch in loader:
            spatial = batch['spatial'].to(device)
            global_feat = batch['global'].to(device)
            mask = batch['mask'].to(device)
            target_policy = batch['policy'].to(device)
            target_value = batch['value'].to(device)
            
            optimizer.zero_grad()
            _, pred_value, logits = model(spatial, global_feat, mask)
            
            # Cross entropy between softmax logits and target policy distribution
            log_probs = torch.log_softmax(logits, dim=-1)
            policy_loss = -torch.sum(target_policy * log_probs) / spatial.size(0)
            value_loss = mse_loss_fn(pred_value, target_value)
            
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_p_loss += policy_loss.item()
            total_v_loss += value_loss.item()
            batches += 1
            
        scheduler.step()
        avg_loss = total_loss / max(batches, 1)
        avg_p = total_p_loss / max(batches, 1)
        avg_v = total_v_loss / max(batches, 1)
        print(f'Epoch {epoch:02d}/{args.epochs:02d} | Loss: {avg_loss:.4f} (Policy: {avg_p:.4f}, Value: {avg_v:.4f}) | lr: {scheduler.get_last_lr()[0]:.6f}')
        
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.out_dir, f'augment_net_epoch_{epoch:03d}.pt')
            torch.save(model.state_dict(), ckpt_path)
            print(f'  Saved checkpoint -> {ckpt_path}')
            
    best_path = os.path.join(args.out_dir, 'augment_net_best.pt')
    torch.save(model.state_dict(), best_path)
    print(f'[Augmentia Trainer] Finished training! Final best model -> {best_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='dataset.jsonl', help='Path to self-play JSONL dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=f32 if hasattr(float, 'f32') else float, default=1e-3, help='Learning rate')
    parser.add_argument('--blocks', type=int, default=6, help='Number of ResNet blocks')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Channels in hidden layers')
    parser.add_argument('--save-freq', type=int, default=5, help='Checkpoint save frequency')
    parser.add_argument('--out-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--cpu', action='store_true', help='Force CPU training')
    args = parser.parse_args()
    train(args)

