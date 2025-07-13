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
import torch.nn.functional as F # Added for focal loss

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
        self.wandb_initialized = False
        
        if offline_mode:
            print("🔄 Starting in offline mode - will sync to wandb later")
            self.wandb_initialized = False
            return
        
        try:
            wandb.init(project=project_name, config=config, settings=wandb.Settings(init_timeout=10))
            self.online = True
            self.wandb_initialized = True
            print("✅ Connected to wandb online")
        except Exception as e:
            print(f"⚠️  Could not connect to wandb: {e}")
            print("📱 Running without wandb - logs will be saved locally only")
            self.online = False
            self.wandb_initialized = False
    
    def log(self, metrics, step=None, commit=True):
        """Log metrics to wandb and/or local logger"""
        if self.wandb_initialized:
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
        if self.wandb_initialized:
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
                 local_logger=None,
                 gradient_accumulation_steps=1):  # NEW: Gradient accumulation
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.wandb_manager = wandb_manager
        self.local_logger = local_logger
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Move model to device
        self.model.to(device)
        
        # IMPROVED: Better optimizer configuration
        whisper_param_ids = {id(p) for p in self.model.whisper.parameters()}
        other_params = [p for p in self.model.parameters() if id(p) not in whisper_param_ids if p.requires_grad]
        whisper_params = [p for p in self.model.parameters() if id(p) in whisper_param_ids if p.requires_grad]
        
        # Increase learning rates and reduce weight decay for better convergence
        self.optimizer = optim.AdamW([
            {'params': whisper_params, 'lr': lr * 0.1, 'weight_decay': weight_decay * 0.5},  # Higher LR for Whisper
            {'params': other_params, 'lr': lr, 'weight_decay': weight_decay}  # Standard LR for other parts
        ], betas=(0.9, 0.98), eps=1e-6)  # Better beta values for Transformer training
        
        # IMPROVED: More aggressive learning rate scheduler for faster convergence
        total_steps = len(train_dataloader) * 50 // gradient_accumulation_steps  # Estimate total steps
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=total_steps // 10,  # Restart every 10% of training
            T_mult=1,
            eta_min=lr * 0.01,  # Minimum LR
            last_epoch=-1
        )
        
        # IMPROVED: Focal loss for better handling of class imbalance
        self.note_loss_fn = self.focal_loss  # Use focal loss instead of CrossEntropy
        self.time_loss_fn = nn.CrossEntropyLoss(ignore_index=model.pad_token, label_smoothing=0.05)  # Reduced label smoothing
        self.duration_loss_fn = nn.CrossEntropyLoss(ignore_index=model.pad_token, label_smoothing=0.05)
        
        # Metrics tracking with better early stopping
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_note_accuracy = 0.0  # NEW: Track best accuracy too
        self.patience_counter = 0
        self.min_improvement = 0.001  # Minimum improvement to reset patience
        
    def focal_loss(self, logits, targets, alpha=1.0, gamma=2.0):
        """Focal loss for better handling of class imbalance"""
        ce_loss = F.cross_entropy(logits, targets, ignore_index=self.model.pad_token, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def prepare_batch(self, batch):
        """Prepare batch for training with data augmentation"""
        audio_features = batch['audio_features'].to(self.device)
        notes_list = batch['notes']
        
        # IMPROVED: Less aggressive audio augmentation
        if self.model.training and np.random.random() < 0.3:
            # Add very small amount of noise to audio features (reduced)
            noise = torch.randn_like(audio_features) * 0.005
            audio_features = audio_features + noise
        
        # Tokenize notes for each sample in batch
        tokenized_notes = []
        max_len = 0
        
        for notes in notes_list:
            # IMPROVED: Less aggressive data augmentation
            if self.model.training and len(notes) > 4 and np.random.random() < 0.15:
                # Randomly drop 5% of notes for robustness (reduced from 10%)
                keep_indices = np.random.choice(len(notes), int(len(notes) * 0.95), replace=False)
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
        """Compute combined loss with improved weighting"""
        batch_size, seq_len = target_tokens.shape
        
        # Shift targets for next-token prediction
        input_tokens = target_tokens[:, :-1]  # Input: all but last
        target_tokens = target_tokens[:, 1:]  # Target: all but first
        
        # Only use note prediction for now (simplified)
        note_logits = note_logits[:, :-1]  # Match target length
        
        # Flatten for loss computation
        note_logits_flat = note_logits.reshape(-1, note_logits.size(-1))
        targets_flat = target_tokens.reshape(-1)
        
        # IMPROVED: Use focal loss for better learning
        loss = self.focal_loss(note_logits_flat, targets_flat)
        
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
        """Train for one epoch with gradient accumulation"""
        self.model.train()
        total_loss = 0
        total_metrics = {'token_accuracy': 0, 'note_accuracy': 0, 'pitch_tolerance_acc': 0, 'perplexity': 0}
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        
        # IMPROVED: Gradient accumulation for more stable training
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Prepare batch
                audio_features, target_tokens = self.prepare_batch(batch)
                
                # Forward pass
                note_logits, time_logits, duration_logits = self.model(
                    audio_features, target_tokens
                )
                
                # Compute loss
                loss = self.compute_loss(note_logits, time_logits, duration_logits, target_tokens)
                
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Gradient step every accumulation_steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                # Compute metrics
                metrics = self.compute_metrics(note_logits, time_logits, duration_logits, target_tokens)
                
                # Update totals
                total_loss += loss.item() * self.gradient_accumulation_steps  # Unscale for reporting
                for key in total_metrics:
                    total_metrics[key] += metrics[key]
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                    'note_acc': f"{metrics['note_accuracy']:.4f}",
                    'pitch_acc': f"{metrics['pitch_tolerance_acc']:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
                
                # Log to wandb and local logger
                if batch_idx % (10 * self.gradient_accumulation_steps) == 0:
                    log_metrics = {
                        'train/loss': loss.item() * self.gradient_accumulation_steps,
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
        
        # Final gradient step if needed
        if len(self.train_dataloader) % self.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
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
    
    def train(self, num_epochs, checkpoint_dir='checkpoints', save_every=5, patience=15):
        """Main training loop with improved early stopping"""
        print(f"Starting training for {num_epochs} epochs")
        print(f"Training samples: {len(self.train_dataloader.dataset)}")
        print(f"Validation samples: {len(self.val_dataloader.dataset)}")
        
        # IMPROVED: More flexible early stopping criteria
        best_combined_metric = 0.0  # Combination of low loss and high accuracy
        no_improvement_epochs = 0
        max_patience = patience  # Configurable patience
        min_epochs = 3  # Minimum epochs before early stopping
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.validate(epoch)
            self.val_losses.append(val_loss)
            
            # IMPROVED: Combined metric for better early stopping
            # Balance between low loss and high accuracy
            combined_metric = val_metrics['note_accuracy'] - (val_loss * 0.1)
            
            # Check for improvement
            improved = False
            if val_loss < self.best_val_loss - self.min_improvement:
                self.best_val_loss = val_loss
                improved = True
            
            if val_metrics['note_accuracy'] > self.best_note_accuracy + self.min_improvement:
                self.best_note_accuracy = val_metrics['note_accuracy']
                improved = True
            
            if combined_metric > best_combined_metric + self.min_improvement:
                best_combined_metric = combined_metric
                improved = True
            
            if improved:
                no_improvement_epochs = 0
                print(f"📈 Improvement detected - resetting patience")
            else:
                no_improvement_epochs += 1
            
            # Print epoch summary
            print(f"Epoch {epoch}: "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Note Acc: {train_metrics['note_accuracy']:.4f}/{val_metrics['note_accuracy']:.4f}, "
                  f"Pitch Acc: {train_metrics['pitch_tolerance_acc']:.4f}/{val_metrics['pitch_tolerance_acc']:.4f}, "
                  f"No improvement: {no_improvement_epochs}/{max_patience}")
            
            # Log epoch summary to local logger
            if self.local_logger:
                self.local_logger.log_epoch_summary(epoch, train_loss, val_loss, train_metrics, val_metrics)
            
            # Save checkpoint
            if epoch % save_every == 0 or epoch == num_epochs - 1 or improved:
                self.save_checkpoint(epoch, val_loss, checkpoint_dir)
            
            # IMPROVED: More intelligent early stopping
            if epoch >= min_epochs and no_improvement_epochs >= max_patience:
                print(f"🛑 Early stopping at epoch {epoch}")
                print(f"   No improvement for {no_improvement_epochs} epochs")
                print(f"   Best val loss: {self.best_val_loss:.4f}")
                print(f"   Best note accuracy: {self.best_note_accuracy:.4f}")
                break
            
            # Additional stopping criteria for very poor performance
            if epoch >= 5 and val_metrics['note_accuracy'] < 0.05:
                print(f"🛑 Stopping due to poor performance (note accuracy < 5%)")
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
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for regularization')
    parser.add_argument('--grad_accum_steps', type=int, default=2,
                        help='Gradient accumulation steps')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    
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
    
    # Create trainer with improved settings
    trainer = VoiceToNotesTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=0.5,  # Reduced gradient clipping for better learning
        wandb_manager=wandb_manager,
        local_logger=local_logger,
        gradient_accumulation_steps=args.grad_accum_steps
    )
    
    # Train
    trainer.train(
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        save_every=1,
        patience=args.patience
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