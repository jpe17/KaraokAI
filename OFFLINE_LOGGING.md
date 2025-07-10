# Offline Logging for KaraokeAI Training

This system allows you to train your model offline and sync results to wandb later when you're back online.

## Features

- **Dual Logging**: Always saves logs locally as JSON files, plus wandb when online
- **Automatic Offline Detection**: Gracefully handles network issues
- **Easy Sync**: Simple commands to sync logs when back online
- **Backup Safety**: Local logs serve as backup even when wandb is working

## Usage

### 1. Training Offline

```bash
# Train in offline mode - saves logs locally
python train.py --offline --fast_dev

# Or disable wandb entirely
python train.py --no_wandb --fast_dev
```

### 2. Training with Automatic Fallback

```bash
# Normal training - automatically falls back to offline if wandb fails
python train.py --fast_dev
```

### 3. Syncing Logs Later

When you're back online, you have several options:

#### Option A: Built-in sync command
```bash
# Sync using the built-in command
python train.py --sync_logs --wandb_project voice-to-notes
```

#### Option B: Dedicated sync script
```bash
# Sync local JSON logs to wandb
python sync_logs.py --project voice-to-notes --log_dir logs

# Also sync wandb offline runs
python sync_logs.py --project voice-to-notes --sync_wandb_offline
```

#### Option C: Manual wandb sync (for wandb offline runs only)
```bash
# Sync wandb offline runs
wandb sync --include-offline
```

## File Structure

```
logs/
├── training_log_20240115_143022.json    # Local training logs
├── training_log_20240115_150845.json    # Another training session
└── ...

~/.local/share/wandb/
├── offline-run-20240115_143022-abc123/  # wandb offline runs
└── ...
```

## Local Log Format

Local logs are saved as JSON files with this structure:

```json
{
  "config": {
    "batch_size": 8,
    "learning_rate": 0.0001,
    "epochs": 5,
    "model": "VoiceToNotesModel",
    "encoder": "whisper-base"
  },
  "train_metrics": [
    {
      "timestamp": "2024-01-15T14:30:22",
      "step": 10,
      "epoch": 0,
      "train/loss": 2.4567,
      "train/note_accuracy": 0.1234,
      "train/pitch_tolerance_acc": 0.2345
    }
  ],
  "val_metrics": [
    {
      "timestamp": "2024-01-15T14:35:22",
      "epoch": 0,
      "val/loss": 2.3456,
      "val/note_accuracy": 0.1345,
      "val/pitch_tolerance_acc": 0.2456
    }
  ],
  "epoch_summaries": [
    {
      "epoch": 0,
      "timestamp": "2024-01-15T14:35:22",
      "train_loss": 2.4567,
      "val_loss": 2.3456,
      "train_metrics": {"note_accuracy": 0.1234, "pitch_tolerance_acc": 0.2345},
      "val_metrics": {"note_accuracy": 0.1345, "pitch_tolerance_acc": 0.2456}
    }
  ]
}
```

## Benefits

1. **Never Lose Data**: Local logs ensure you never lose training metrics
2. **Flexible Syncing**: Sync when convenient, not during training
3. **Offline Development**: Develop and test models anywhere
4. **Backup**: Local logs serve as backup even when wandb is working
5. **Batch Syncing**: Sync multiple training sessions at once

## Tips

- Use `--fast_dev` for quick testing offline
- Local logs are timestamped, so you can run multiple sessions
- The sync script creates separate wandb runs for each local log file
- You can customize run names when syncing: `--run_name "my_experiment"`

## Troubleshooting

**Q: Training fails with wandb error**
A: The system should automatically fall back to offline mode. If not, use `--offline` flag.

**Q: Sync script fails**
A: Check your internet connection and wandb login status: `wandb login`

**Q: Multiple log files, which to sync?**
A: The sync script processes all log files in the directory. Each becomes a separate wandb run.

**Q: Want to sync only specific experiments?**
A: Move the desired log files to a separate directory and sync that directory. 