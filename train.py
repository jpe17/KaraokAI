import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import wandb
import os
import json
from tqdm import tqdm
import argparse
from datetime import datetime
import time

from model import VoiceToNotesModel
from dataloader import create_dataloaders

class LocalLogger:
    """Local logger that saves metrics to JSON files"""
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f'training_log_{timestamp}.json')
        
        # Initialize log structure
        self.logs = {
            'config': {},
            'train_metrics': [],
            'val_metrics': [],
            'epoch_summaries': []
        }
        
        print(f"Local logging to: {self.log_file}")
    
    def log_config(self, config):
        """Log training configuration"""
        self.logs['config'] = config
        self._save_logs()
    
    def log_metrics(self, metrics, step=None, epoch=None):
        """Log training/validation metrics"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'epoch': epoch,
            **metrics
        }
        
        # Separate train/val metrics
        if any(key.startswith('train/') for key in metrics.keys()):
            self.logs['train_metrics'].append(log_entry)
        elif any(key.startswith('val/') for key in metrics.keys()):
            self.logs['val_metrics'].append(log_entry)
        
        self._save_logs()
    
    def log_epoch_summary(self, epoch, train_loss, val_loss, train_metrics, val_metrics):
        """Log epoch summary"""
        summary = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
        
        self.logs['epoch_summaries'].append(summary)
        self._save_logs()
    
    def _save_logs(self):
        """Save logs to JSON file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.logs, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save local logs: {e}")
    
    def sync_to_wandb(self):
        """Sync local logs to wandb (call this when back online)"""
        if not self.logs['train_metrics'] and not self.logs['val_metrics']:
            print("No local logs to sync")
            return
        
        print(f"Syncing {len(self.logs['train_metrics'])} train metrics and {len(self.logs['val_metrics'])} val metrics to wandb...")
        
        # Sync train metrics
        for entry in self.logs['train_metrics']:
            metrics = {k: v for k, v in entry.items() if k not in ['timestamp', 'step', 'epoch']}
            wandb.log(metrics, step=entry.get('step'), commit=False)
        
        # Sync val metrics
        for entry in self.logs['val_metrics']:
            metrics = {k: v for k, v in entry.items() if k not in ['timestamp', 'step', 'epoch']}
            wandb.log(metrics, step=entry.get('step'), commit=True)
        
        print("✅ Local logs synced to wandb!")

class WandbManager:
    """Manages wandb connection with offline fallback"""
    def __init__(self, project_name, config, offline_mode=False, local_logger=None):
        self.project_name = project_name
        self.config = config
        self.local_logger = local_logger
        self.online = False
        self.offline_mode = offline_mode
        
        if offline_mode:
            print("🔄 Starting in offline mode - will sync to wandb later")
            os.environ["WANDB_MODE"] = "offline"
        
        try:
            wandb.init(project=project_name, config=config)
            self.online = True
            print("✅ Connected to wandb online")
        except Exception as e:
            print(f"⚠️  Could not connect to wandb: {e}")
            print("📱 Running in offline mode - logs will be saved locally")
            self.online = False
            # Switch to offline mode
            os.environ["WANDB_MODE"] = "offline"
            wandb.init(project=project_name, config=config)
    
    def log(self, metrics, step=None, commit=True):
        """Log metrics to wandb and/or local logger"""
        try:
            wandb.log(metrics, step=step, commit=commit)
            if not self.online:
                print("📝 Logged to wandb offline cache")
        except Exception as e:
            print(f"⚠️  wandb logging failed: {e}")
        
        # Always log locally as backup
        if self.local_logger:
            self.local_logger.log_metrics(metrics, step=step)
    
    def finish(self):
        """Finish wandb run"""
        try:
            wandb.finish()
        except Exception as e:
            print(f"Warning: Could not finish wandb run: {e}")
    
    def sync_offline_runs(self):
        """Sync offline wandb runs (call this when back online)"""
        try:
            import subprocess
            result = subprocess.run(['wandb', 'sync', '--include-offline'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Offline wandb runs synced successfully!")
            else:
                print(f"⚠️  wandb sync failed: {result.stderr}")
        except Exception as e:
            print(f"⚠️  Could not sync offline runs: {e}")

class VoiceToNotesTrainer:
    def __init__(self, 
                 model,
                 train_dataloader,
                 val_dataloader,
                 device,
                 lr=1e-4,
                 weight_decay=1e-5,
                 warmup_steps=1000,
                 max_grad_norm=1.0,
                 wandb_manager=None,
                 local_logger=None):
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.wandb_manager = wandb_manager
        self.local_logger = local_logger
        
        # Move model to device
        self.model.to(device)
        
        # Optimizer with different learning rates for different parts
        whisper_param_ids = {id(p) for p in self.model.whisper.parameters()}
        other_params = [p for p in self.model.parameters() if id(p) not in whisper_param_ids]
        whisper_params = [p for p in self.model.parameters() if id(p) in whisper_param_ids]
        
        self.optimizer = optim.AdamW([
            {'params': whisper_params, 'lr': lr * 0.05},  # Even lower LR for pretrained Whisper
            {'params': other_params, 'lr': lr * 0.5}  # Reduced LR for other params too
        ], weight_decay=weight_decay * 2)  # Increased weight decay
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr,
            steps_per_epoch=len(train_dataloader),
            epochs=100,  # Will be updated based on actual training
            pct_start=0.1
        )
        
        # Loss functions with label smoothing to reduce overfitting
        self.note_loss_fn = nn.CrossEntropyLoss(ignore_index=model.pad_token, label_smoothing=0.1)
        self.time_loss_fn = nn.CrossEntropyLoss(ignore_index=model.pad_token, label_smoothing=0.1)
        self.duration_loss_fn = nn.CrossEntropyLoss(ignore_index=model.pad_token, label_smoothing=0.1)
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def prepare_batch(self, batch):
        """Prepare batch for training with data augmentation"""
        audio_features = batch['audio_features'].to(self.device)
        notes_list = batch['notes']
        
        # Apply audio augmentation during training
        if self.model.training:
            # Add small amount of noise to audio features
            noise = torch.randn_like(audio_features) * 0.01
            audio_features = audio_features + noise
        
        # Tokenize notes for each sample in batch
        tokenized_notes = []
        max_len = 0
        
        for notes in notes_list:
            # Apply note sequence augmentation during training
            if self.model.training and len(notes) > 2 and np.random.random() < 0.3:
                # Randomly drop 10% of notes for robustness
                keep_indices = np.random.choice(len(notes), int(len(notes) * 0.9), replace=False)
                notes = [notes[i] for i in sorted(keep_indices)]
            
            tokens = self.model.tokenize_notes(notes)
            tokenized_notes.append(tokens)
            max_len = max(max_len, len(tokens))
        
        # Pad sequences
        padded_notes = []
        for tokens in tokenized_notes:
            padded = tokens + [self.model.pad_token] * (max_len - len(tokens))
            padded_notes.append(padded)
        
        target_tokens = torch.tensor(padded_notes, dtype=torch.long).to(self.device)
        
        return audio_features, target_tokens
    
    def compute_loss(self, note_logits, time_logits, duration_logits, target_tokens):
        """Compute combined loss"""
        batch_size, seq_len = target_tokens.shape
        
        # For now, use simplified loss - just predict next note
        # In a more sophisticated version, you'd separate note/time/duration targets
        
        # Shift targets for next-token prediction
        input_tokens = target_tokens[:, :-1]  # Input: all but last
        target_tokens = target_tokens[:, 1:]  # Target: all but first
        
        # Only use note prediction for now (simplified)
        note_logits = note_logits[:, :-1]  # Match target length
        
        # Flatten for loss computation
        note_logits_flat = note_logits.reshape(-1, note_logits.size(-1))
        targets_flat = target_tokens.reshape(-1)
        
        # Compute loss
        loss = self.note_loss_fn(note_logits_flat, targets_flat)
        
        return loss
    
    def compute_metrics(self, note_logits, time_logits, duration_logits, target_tokens):
        """Compute evaluation metrics"""
        with torch.no_grad():
            # Token-level accuracy (existing)
            note_preds = torch.argmax(note_logits[:, :-1], dim=-1)
            targets = target_tokens[:, 1:]
            
            # Mask out padding tokens
            mask = targets != self.model.pad_token
            correct = (note_preds == targets) & mask
            token_accuracy = correct.sum().float() / mask.sum().float()
            
            # Perplexity
            loss = self.compute_loss(note_logits, time_logits, duration_logits, target_tokens)
            perplexity = torch.exp(loss)
            
            # NEW: Musical accuracy metrics
            musical_metrics = self.compute_musical_metrics(note_preds, targets, mask)
            
            metrics = {
                'token_accuracy': token_accuracy.item(),  # Renamed for clarity
                'note_accuracy': musical_metrics['note_accuracy'],  # NEW: Just note correctness
                'pitch_tolerance_acc': musical_metrics['pitch_tolerance_acc'],  # NEW: ±1 semitone tolerance
                'perplexity': perplexity.item(),
                'loss': loss.item()
            }
            
            return metrics
    
    def compute_musical_metrics(self, note_preds, targets, mask):
        """Compute musically meaningful metrics"""
        with torch.no_grad():
            # Only look at valid (non-padded) positions
            valid_preds = note_preds[mask]
            valid_targets = targets[mask]
            
            # Convert tokens back to note numbers (subtract special token offset)
            # Only consider tokens that are actually notes (3-130 range)
            note_mask = (valid_targets >= 3) & (valid_targets < 131)
            if note_mask.sum() == 0:
                return {'note_accuracy': 0.0, 'pitch_tolerance_acc': 0.0}
            
            pred_notes = valid_preds[note_mask] - 3  # Convert to MIDI note numbers
            target_notes = valid_targets[note_mask] - 3
            
            # Exact note accuracy (ignoring timing)
            note_accuracy = (pred_notes == target_notes).float().mean()
            
            # Pitch tolerance accuracy (±1 semitone)
            pitch_diff = torch.abs(pred_notes - target_notes)
            pitch_tolerance_acc = (pitch_diff <= 1).float().mean()
            
            return {
                'note_accuracy': note_accuracy.item(),
                'pitch_tolerance_acc': pitch_tolerance_acc.item()
            }
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_metrics = {'token_accuracy': 0, 'note_accuracy': 0, 'pitch_tolerance_acc': 0, 'perplexity': 0}
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Prepare batch
                audio_features, target_tokens = self.prepare_batch(batch)
                
                # Forward pass
                self.optimizer.zero_grad()
                note_logits, time_logits, duration_logits = self.model(
                    audio_features, target_tokens
                )
                
                # Compute loss
                loss = self.compute_loss(note_logits, time_logits, duration_logits, target_tokens)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                
                # Compute metrics
                metrics = self.compute_metrics(note_logits, time_logits, duration_logits, target_tokens)
                
                # Update totals
                total_loss += loss.item()
                for key in total_metrics:
                    total_metrics[key] += metrics[key]
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'note_acc': f"{metrics['note_accuracy']:.4f}",
                    'pitch_acc': f"{metrics['pitch_tolerance_acc']:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
                
                # Log to wandb and local logger
                if batch_idx % 10 == 0:
                    log_metrics = {
                        'train/loss': loss.item(),
                        'train/token_accuracy': metrics['token_accuracy'],
                        'train/note_accuracy': metrics['note_accuracy'],
                        'train/pitch_tolerance_acc': metrics['pitch_tolerance_acc'],
                        'train/perplexity': metrics['perplexity'],
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'epoch': epoch,
                        'step': epoch * len(self.train_dataloader) + batch_idx
                    }
                    
                    if self.wandb_manager:
                        self.wandb_manager.log(log_metrics, step=log_metrics['step'])
                    elif self.local_logger:
                        self.local_logger.log_metrics(log_metrics, step=log_metrics['step'], epoch=epoch)
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Average metrics
        avg_loss = total_loss / len(self.train_dataloader)
        avg_metrics = {k: v / len(self.train_dataloader) for k, v in total_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_metrics = {'token_accuracy': 0, 'note_accuracy': 0, 'pitch_tolerance_acc': 0, 'perplexity': 0}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_dataloader, desc="Validation")):
                try:
                    # Prepare batch
                    audio_features, target_tokens = self.prepare_batch(batch)
                    
                    # Forward pass
                    note_logits, time_logits, duration_logits = self.model(
                        audio_features, target_tokens
                    )
                    
                    # Compute loss and metrics
                    loss = self.compute_loss(note_logits, time_logits, duration_logits, target_tokens)
                    metrics = self.compute_metrics(note_logits, time_logits, duration_logits, target_tokens)
                    
                    # Update totals
                    total_loss += loss.item()
                    for key in total_metrics:
                        total_metrics[key] += metrics[key]
                        
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        # Average metrics
        avg_loss = total_loss / len(self.val_dataloader)
        avg_metrics = {k: v / len(self.val_dataloader) for k, v in total_metrics.items()}
        
        # Log to wandb and local logger
        log_metrics = {
            'val/loss': avg_loss,
            'val/token_accuracy': avg_metrics['token_accuracy'],
            'val/note_accuracy': avg_metrics['note_accuracy'],
            'val/pitch_tolerance_acc': avg_metrics['pitch_tolerance_acc'],
            'val/perplexity': avg_metrics['perplexity'],
            'epoch': epoch
        }
        
        if self.wandb_manager:
            self.wandb_manager.log(log_metrics)
        elif self.local_logger:
            self.local_logger.log_metrics(log_metrics, epoch=epoch)
        
        return avg_loss, avg_metrics
    
    def save_checkpoint(self, epoch, val_loss, checkpoint_dir='checkpoints'):
        """Save model checkpoint"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        # Save current checkpoint
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'))
        
        # Save best checkpoint
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_checkpoint.pt'))
            print(f"New best checkpoint saved with val_loss: {val_loss:.4f}")
    
    def train(self, num_epochs, checkpoint_dir='checkpoints', save_every=5):
        """Main training loop"""
        print(f"Starting training for {num_epochs} epochs")
        print(f"Training samples: {len(self.train_dataloader.dataset)}")
        print(f"Validation samples: {len(self.val_dataloader.dataset)}")
        
        patience_counter = 0  # Initialize patience counter
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.validate(epoch)
            self.val_losses.append(val_loss)
            
            # Print epoch summary
            print(f"Epoch {epoch}: "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Note Acc: {train_metrics['note_accuracy']:.4f}/{val_metrics['note_accuracy']:.4f}, "
                  f"Pitch Acc: {train_metrics['pitch_tolerance_acc']:.4f}/{val_metrics['pitch_tolerance_acc']:.4f}")
            
            # Log epoch summary to local logger
            if self.local_logger:
                self.local_logger.log_epoch_summary(epoch, train_loss, val_loss, train_metrics, val_metrics)
            
            # Save checkpoint
            if epoch % save_every == 0 or epoch == num_epochs - 1:
                self.save_checkpoint(epoch, val_loss, checkpoint_dir)
            
            # Early stopping with more aggressive patience for overfitting
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 5:  # More aggressive early stopping
                    print(f"Early stopping at epoch {epoch} due to overfitting")
                    break

def main():
    parser = argparse.ArgumentParser(description='Train Voice-to-Notes Model')
    parser.add_argument('--voices_dir', type=str, default='processed_data/voices',
                        help='Directory containing voice files')
    parser.add_argument('--notes_dir', type=str, default='processed_data/notes',
                        help='Directory containing note files')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--wandb_project', type=str, default='voice-to-notes',
                        help='Weights & Biases project name')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--offline', action='store_true',
                        help='Run in offline mode (save logs locally)')
    parser.add_argument('--sync_logs', action='store_true',
                        help='Sync offline logs to wandb and exit')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save local logs')
    # NEW: Fast development options
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit number of samples for faster development')
    parser.add_argument('--cache_features', action='store_true',
                        help='Cache audio features for faster loading')
    parser.add_argument('--fast_dev', action='store_true',
                        help='Enable fast development mode (smaller model, fewer epochs)')
    
    args = parser.parse_args()
    
    # Handle log syncing mode
    if args.sync_logs:
        print("🔄 Syncing offline logs to wandb...")
        
        # Initialize local logger to find log files
        local_logger = LocalLogger(args.log_dir)
        
        # Sync wandb offline runs
        wandb_manager = WandbManager(args.wandb_project, {})
        wandb_manager.sync_offline_runs()
        
        # Also sync from local JSON logs if available
        log_files = [f for f in os.listdir(args.log_dir) if f.startswith('training_log_') and f.endswith('.json')]
        if log_files:
            print(f"Found {len(log_files)} local log files to sync")
            for log_file in log_files:
                print(f"Would sync {log_file} (implement if needed)")
        
        print("✅ Sync complete!")
        return
    
    # Initialize logging
    config = {
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'model': 'VoiceToNotesModel',
        'encoder': 'whisper-base',
        'max_samples': args.max_samples,
        'fast_dev': args.fast_dev,
        'offline_mode': args.offline
    }
    
    # Always create local logger as backup
    local_logger = LocalLogger(args.log_dir)
    local_logger.log_config(config)
    
    # Initialize wandb manager
    wandb_manager = None
    if not args.no_wandb:
        wandb_manager = WandbManager(
            args.wandb_project,
            config,
            offline_mode=args.offline,
            local_logger=local_logger
        )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # NEW: Fast development mode
    if args.fast_dev:
        print("🚀 Fast development mode enabled!")
        args.max_samples = args.max_samples or 20  # Default to 20 samples
        args.epochs = min(args.epochs, 5)  # Max 5 epochs
        args.batch_size = min(args.batch_size, 4)  # Smaller batch size
        print(f"Fast dev settings: {args.max_samples} samples, {args.epochs} epochs, batch_size={args.batch_size}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_dataloader, val_dataloader = create_dataloaders(
        args.voices_dir,
        args.notes_dir,
        batch_size=args.batch_size,
        train_split=0.8,
        max_audio_length=10.0,
        max_notes=100,
        max_samples=args.max_samples,  # NEW: Pass sample limit
        cache_features=args.cache_features  # NEW: Pass cache option
    )
    
    # Create model
    print("Creating model...")
    model = VoiceToNotesModel(
        whisper_model_name="openai/whisper-base",
        d_model=512,
        nhead=8,
        num_decoder_layers=6,
        max_notes=100
    )
    
    # Create trainer
    trainer = VoiceToNotesTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        lr=args.lr,
        wandb_manager=wandb_manager,
        local_logger=local_logger
    )
    
    # Train
    trainer.train(
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        save_every=5
    )
    
    # Finish wandb run
    if wandb_manager:
        wandb_manager.finish()
    
    print("Training completed!")
    print(f"📁 Local logs saved to: {local_logger.log_file}")
    
    if args.offline:
        print("🔄 To sync logs when back online, run:")
        print(f"python train.py --sync_logs --log_dir {args.log_dir} --wandb_project {args.wandb_project}")

if __name__ == "__main__":
    main() 