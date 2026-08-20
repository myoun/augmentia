import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import AugmentNet
from dataset import AugmentDataset

torch.backends.cudnn.benchmark = True

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f'[Augmentia Trainer] Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"})')

    dataset = AugmentDataset(
        primary_jsonl=args.data,
        replay_jsonl=args.replay_data,
        replay_ratio=args.replay_ratio,
        augment_flip=not args.no_flip
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    model = AugmentNet(num_blocks=args.blocks, hidden_dim=args.hidden_dim).to(device)

    # Load existing checkpoint for continuous fine-tuning
    if args.checkpoint and os.path.isfile(args.checkpoint):
        try:
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
            print(f'[Augmentia Trainer] Continuous Learning: Loaded weights from {args.checkpoint}')
        except Exception as e:
            print(f'[Augmentia Trainer] Warning: Failed to load checkpoint {args.checkpoint}: {e}')
    elif os.path.isfile(os.path.join(args.out_dir, 'augment_net_best.pt')) and not args.fresh:
        auto_ckpt = os.path.join(args.out_dir, 'augment_net_best.pt')
        try:
            model.load_state_dict(torch.load(auto_ckpt, map_location=device))
            print(f'[Augmentia Trainer] Auto-resume: Loaded previous best weights from {auto_ckpt}')
        except Exception as e:
            print(f'[Augmentia Trainer] Warning: Failed auto-resume: {e}')
    else:
        print('[Augmentia Trainer] Starting fresh model training from scratch')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    mse_loss_fn = nn.MSELoss()
    os.makedirs(args.out_dir, exist_ok=True)

    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    print(f'[Augmentia Trainer] Starting training for {args.epochs} epochs with batch size {args.batch_size} (FP16 AMP: {scaler is not None})...')
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_p_loss = 0.0
        total_v_loss = 0.0
        batches = 0

        for batch in loader:
            spatial = batch['spatial'].to(device, non_blocking=True)
            global_feat = batch['global'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)
            target_policy = batch['policy'].to(device, non_blocking=True)
            target_value = batch['value'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if scaler:
                with torch.amp.autocast('cuda'):
                    _, pred_value, logits = model(spatial, global_feat, mask)
                    log_probs = torch.log_softmax(logits, dim=-1)
                    policy_loss = -torch.sum(target_policy * log_probs) / spatial.size(0)
                    value_loss = mse_loss_fn(pred_value, target_value)
                    loss = policy_loss + value_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                _, pred_value, logits = model(spatial, global_feat, mask)
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

    best_path = os.path.join(args.out_dir, 'augment_net_best.pt')
    torch.save(model.state_dict(), best_path)
    print(f'[Augmentia Trainer] Finished! Updated best model -> {best_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='dataset.jsonl', help='Path to primary JSONL dataset')
    parser.add_argument('--replay-data', type=str, default='', help='Path to replay buffer dataset')
    parser.add_argument('--replay-ratio', type=float, default=0.0, help='Ratio of replay buffer samples')
    parser.add_argument('--checkpoint', type=str, default='', help='Path to initial weights (.pt) for fine-tuning')
    parser.add_argument('--fresh', action='store_true', help='Force training from scratch without loading previous weights')
    parser.add_argument('--no-flip', action='store_true', help='Disable horizontal flip data augmentation')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--blocks', type=int, default=6, help='Number of ResNet blocks')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Channels in hidden layers')
    parser.add_argument('--save-freq', type=int, default=5, help='Checkpoint save frequency')
    parser.add_argument('--out-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--cpu', action='store_true', help='Force CPU training')
    args = parser.parse_args()
    train(args)
