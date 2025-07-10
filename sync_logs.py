#!/usr/bin/env python3
"""
Sync offline training logs to wandb
"""

import os
import json
import argparse
import wandb
from glob import glob
from datetime import datetime

def sync_local_logs_to_wandb(log_dir, project_name, run_name=None):
    """Sync local JSON logs to wandb"""
    
    # Find all log files
    log_files = glob(os.path.join(log_dir, "training_log_*.json"))
    
    if not log_files:
        print(f"No log files found in {log_dir}")
        return
    
    print(f"Found {len(log_files)} log files to sync")
    
    for log_file in log_files:
        print(f"\n🔄 Syncing {log_file}...")
        
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
            
            # Extract timestamp from filename for run name
            timestamp = os.path.basename(log_file).replace('training_log_', '').replace('.json', '')
            sync_run_name = run_name or f"sync_{timestamp}"
            
            # Initialize wandb run
            wandb.init(
                project=project_name,
                name=sync_run_name,
                config=logs.get('config', {}),
                resume="never"
            )
            
            # Sync train metrics
            print(f"  📊 Syncing {len(logs.get('train_metrics', []))} train metrics...")
            for entry in logs.get('train_metrics', []):
                metrics = {k: v for k, v in entry.items() 
                          if k not in ['timestamp', 'step', 'epoch']}
                step = entry.get('step')
                wandb.log(metrics, step=step, commit=False)
            
            # Sync validation metrics
            print(f"  📊 Syncing {len(logs.get('val_metrics', []))} validation metrics...")
            for entry in logs.get('val_metrics', []):
                metrics = {k: v for k, v in entry.items() 
                          if k not in ['timestamp', 'step', 'epoch']}
                step = entry.get('step')
                wandb.log(metrics, step=step, commit=True)
            
            # Log epoch summaries as a table
            if logs.get('epoch_summaries'):
                print(f"  📊 Syncing {len(logs['epoch_summaries'])} epoch summaries...")
                
                # Create summary table
                summary_data = []
                for summary in logs['epoch_summaries']:
                    summary_data.append([
                        summary['epoch'],
                        summary['train_loss'],
                        summary['val_loss'],
                        summary['train_metrics'].get('note_accuracy', 0),
                        summary['val_metrics'].get('note_accuracy', 0),
                        summary['train_metrics'].get('pitch_tolerance_acc', 0),
                        summary['val_metrics'].get('pitch_tolerance_acc', 0)
                    ])
                
                table = wandb.Table(
                    columns=["Epoch", "Train Loss", "Val Loss", 
                            "Train Note Acc", "Val Note Acc", 
                            "Train Pitch Acc", "Val Pitch Acc"],
                    data=summary_data
                )
                wandb.log({"epoch_summaries": table})
            
            wandb.finish()
            print(f"  ✅ Successfully synced {log_file}")
            
        except Exception as e:
            print(f"  ❌ Error syncing {log_file}: {e}")
            continue

def sync_wandb_offline_runs():
    """Sync wandb offline runs"""
    import subprocess
    
    try:
        print("🔄 Syncing wandb offline runs...")
        result = subprocess.run(['wandb', 'sync', '--include-offline'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Offline wandb runs synced successfully!")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"⚠️  wandb sync failed: {result.stderr}")
    except Exception as e:
        print(f"⚠️  Could not sync offline runs: {e}")

def main():
    parser = argparse.ArgumentParser(description='Sync offline training logs to wandb')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory containing log files')
    parser.add_argument('--project', type=str, required=True,
                        help='wandb project name')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Custom run name (default: auto-generated)')
    parser.add_argument('--sync_wandb_offline', action='store_true',
                        help='Also sync wandb offline runs')
    
    args = parser.parse_args()
    
    # Sync wandb offline runs first
    if args.sync_wandb_offline:
        sync_wandb_offline_runs()
    
    # Sync local JSON logs
    sync_local_logs_to_wandb(args.log_dir, args.project, args.run_name)
    
    print("\n🎉 Sync complete!")

if __name__ == "__main__":
    main() 