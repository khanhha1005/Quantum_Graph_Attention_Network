"""
Train Quantum Graph Attention Network (QGAT) on vulnerability detection.
Trains only the HEA quantum attention variant (IQP and QAOA removed).

OPTIMIZATIONS:
- Class weighting for imbalanced vulnerability data
- OneCycleLR scheduler for faster convergence
- Gradient clipping for stability
- Reduced qubit count (8 max vs 16) for 4x+ speedup
- Focal loss option for hard examples
- Better default hyperparameters
"""
import os
import sys

# Fix OpenMP conflict on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
import json
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import Counter

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.Quantum_GAT import QGAT
from dataloader.load_vulnerability_data import load_vulnerability_data


# ============================================================================
# FOCAL LOSS - Better for imbalanced data
# ============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Reduces loss for well-classified examples, focuses on hard examples.
    """
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none',
            pos_weight=self.pos_weight
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def compute_class_weights(dataset):
    """Compute class weights for imbalanced data."""
    labels = [int(d.y.item()) for d in dataset]
    counter = Counter(labels)
    total = len(labels)
    
    # Compute weight for positive class (minority)
    n_pos = counter.get(1, 1)
    n_neg = counter.get(0, 1)
    
    # pos_weight = n_neg / n_pos (for BCEWithLogitsLoss)
    pos_weight = n_neg / max(n_pos, 1)
    
    print(f"  Class distribution: {counter}")
    print(f"  Positive class weight: {pos_weight:.2f}")
    
    return torch.tensor([pos_weight])


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=10, min_delta=0.0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute classification metrics including AUC scores."""
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    elif isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    elif isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    if y_prob is not None:
        if isinstance(y_prob, list):
            y_prob = np.array(y_prob)
        elif isinstance(y_prob, torch.Tensor):
            y_prob = y_prob.cpu().numpy()
        y_prob = y_prob.flatten()
    
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='binary', zero_division=0)
    }
    
    # Compute AUC metrics if probabilities are provided
    if y_prob is not None:
        try:
            # ROC-AUC: Area under ROC curve
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            # Handle case where only one class is present
            metrics['roc_auc'] = 0.0
        
        try:
            # PR-AUC: Area under Precision-Recall curve (Average Precision)
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics['pr_auc'] = 0.0
    else:
        metrics['roc_auc'] = 0.0
        metrics['pr_auc'] = 0.0
    
    return metrics


def train_epoch(model, train_loader, criterion, optimizer, device, epoch=None, 
                scheduler=None, grad_clip=1.0):
    """Train for one epoch with gradient clipping and LR scheduling."""
    import time
    model.train()
    total_loss = 0.0
    all_preds = []
    all_probs = []  # Store probabilities for AUC
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        batch = batch.to(device)
        
        if batch_idx == 0:
            start_time = time.time()
        
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        
        if out.dim() > 1 and out.size(1) == 1:
            out = out.squeeze(1)
        
        loss = criterion(out, batch.y.float())
        loss.backward()
        
        # Gradient clipping for stability
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Step LR scheduler if using OneCycleLR
        if scheduler is not None and isinstance(scheduler, OneCycleLR):
            scheduler.step()
        
        probs = torch.sigmoid(out)
        preds = (probs > 0.5).long()
        all_preds.extend(preds.cpu())
        all_probs.extend(probs.detach().cpu())  # Store probabilities
        all_labels.extend(batch.y.cpu())
        total_loss += loss.item()
        
        if batch_idx == 0:
            elapsed = time.time() - start_time
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'time': f'{elapsed:.1f}s'})
        else:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader), compute_metrics(all_labels, all_preds, all_probs)


def validate_epoch(model, val_loader, criterion, device, epoch=None):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []  # Store probabilities for AUC
    all_labels = []
    
    pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch}", leave=False)
    with torch.no_grad():
        for batch in pbar:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            
            if out.dim() > 1 and out.size(1) == 1:
                out = out.squeeze(1)
            
            loss = criterion(out, batch.y.float())
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).long()
            all_preds.extend(preds.cpu())
            all_probs.extend(probs.cpu())  # Store probabilities
            all_labels.extend(batch.y.cpu())
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(val_loader), compute_metrics(all_labels, all_preds, all_probs)


def train_model(vuln_type, attn_model, train_loader, val_loader, config, log_dir=None, 
                pos_weight=None):
    """Train QGAT model with specified attention mechanism - OPTIMIZED."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Training QGAT ({attn_model}) for {vuln_type}")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    writer = SummaryWriter(log_dir=log_dir) if log_dir else None
    
    # Get input dimensions
    sample_batch = next(iter(train_loader))
    input_dims = sample_batch.x.size(1)
    
    # Create model with optimized settings
    model = QGAT(
        input_dims=input_dims,
        q_depths=config['q_depths'],
        output_dims=1,
        attn_model=attn_model,
        dropout=config['dropout'],
        readout=False,
        max_qubits=config.get('max_qubits', 8),  # NEW: configurable max qubits
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: QGAT with {attn_model} attention")
    print(f"Input dims: {input_dims}, Total Parameters: {num_params}, Trainable: {num_trainable_params}")
    print(f"Quantum depths: {config['q_depths']}, Max qubits: {config.get('max_qubits', 8)}")
    print(f"Dropout: {config['dropout']}, Grad clip: {config.get('grad_clip', 1.0)}")
    
    # ================================================================
    # LOSS FUNCTION - with class weighting for imbalance
    # ================================================================
    if config.get('use_focal_loss', False):
        criterion = FocalLoss(
            alpha=config.get('focal_alpha', 0.25),
            gamma=config.get('focal_gamma', 2.0),
            pos_weight=pos_weight.to(device) if pos_weight is not None else None
        )
        print(f"Using Focal Loss (alpha={config.get('focal_alpha', 0.25)}, gamma={config.get('focal_gamma', 2.0)})")
    else:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight.to(device) if pos_weight is not None else None
        )
        if pos_weight is not None:
            print(f"Using BCEWithLogitsLoss with pos_weight={pos_weight.item():.2f}")
    
    # ================================================================
    # OPTIMIZER with weight decay
    # ================================================================
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # ================================================================
    # LEARNING RATE SCHEDULER - OneCycleLR for faster convergence
    # ================================================================
    scheduler = None
    if config.get('use_scheduler', True):
        total_steps = len(train_loader) * config['max_epochs']
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config['learning_rate'] * 10,  # Peak at 10x base LR
            total_steps=total_steps,
            pct_start=0.1,  # 10% warmup
            anneal_strategy='cos',
            div_factor=10,
            final_div_factor=100
        )
        print(f"Using OneCycleLR scheduler (max_lr={config['learning_rate'] * 10})")
    
    early_stopping = EarlyStopping(patience=config['patience'], restore_best_weights=True)
    
    history = {
        'train_loss': [], 'train_accuracy': [], 'train_precision': [], 
        'train_recall': [], 'train_f1': [], 'train_roc_auc': [], 'train_pr_auc': [],
        'val_loss': [], 'val_accuracy': [], 'val_precision': [],
        'val_recall': [], 'val_f1': [], 'val_roc_auc': [], 'val_pr_auc': [], 'lr': []
    }
    
    # Log model info to TensorBoard
    if writer:
        writer.add_scalar('Model/total_parameters', num_params, 0)
        writer.add_scalar('Model/trainable_parameters', num_trainable_params, 0)
        writer.add_scalar('Model/n_qubits', config.get('max_qubits', 8), 0)
        writer.add_scalar('Model/n_layers', config['q_depths'][0], 0)
        
        # Log hyperparameters
        hparams = {
            'attn_model': attn_model,
            'n_qubits': config.get('max_qubits', 8),
            'n_layers': config['q_depths'][0],
            'learning_rate': config['learning_rate'],
            'dropout': config['dropout'],
            'batch_size': config['batch_size'],
            'total_params': num_params,
            'trainable_params': num_trainable_params,
        }
        writer.add_text('Hyperparameters', str(hparams), 0)
    
    best_val_f1 = 0.0
    best_model_state = None
    
    print(f"\nStarting training...")
    epoch_pbar = tqdm(range(1, config['max_epochs'] + 1), desc=f"{vuln_type}-{attn_model}")
    
    for epoch in epoch_pbar:
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            scheduler=scheduler,
            grad_clip=config.get('grad_clip', 1.0)
        )
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        if epoch % config['validate_every'] == 0 or epoch == 1:
            val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device, epoch)
        else:
            val_loss = history['val_loss'][-1] if history['val_loss'] else 0
            val_metrics = {
                'accuracy': history['val_accuracy'][-1] if history['val_accuracy'] else 0,
                'precision': history['val_precision'][-1] if history['val_precision'] else 0,
                'recall': history['val_recall'][-1] if history['val_recall'] else 0,
                'f1': history['val_f1'][-1] if history['val_f1'] else 0,
                'roc_auc': history['val_roc_auc'][-1] if history['val_roc_auc'] else 0,
                'pr_auc': history['val_pr_auc'][-1] if history['val_pr_auc'] else 0
            }
        
        # Store all metrics in history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['train_precision'].append(train_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        history['train_f1'].append(train_metrics['f1'])
        history['train_roc_auc'].append(train_metrics['roc_auc'])
        history['train_pr_auc'].append(train_metrics['pr_auc'])
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_roc_auc'].append(val_metrics['roc_auc'])
        history['val_pr_auc'].append(val_metrics['pr_auc'])
        
        epoch_pbar.set_postfix({
            'loss': f'{train_loss:.4f}', 
            'val_f1': f'{val_metrics["f1"]:.4f}',
            'lr': f'{current_lr:.2e}'
        })
        
        # Log all metrics to TensorBoard
        if writer:
            # Loss
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            
            # Accuracy
            writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
            writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            
            # Precision
            writer.add_scalar('Precision/train', train_metrics['precision'], epoch)
            writer.add_scalar('Precision/val', val_metrics['precision'], epoch)
            
            # Recall
            writer.add_scalar('Recall/train', train_metrics['recall'], epoch)
            writer.add_scalar('Recall/val', val_metrics['recall'], epoch)
            
            # F1 Score
            writer.add_scalar('F1/train', train_metrics['f1'], epoch)
            writer.add_scalar('F1/val', val_metrics['f1'], epoch)
            
            # ROC-AUC
            writer.add_scalar('ROC_AUC/train', train_metrics['roc_auc'], epoch)
            writer.add_scalar('ROC_AUC/val', val_metrics['roc_auc'], epoch)
            
            # PR-AUC (Average Precision)
            writer.add_scalar('PR_AUC/train', train_metrics['pr_auc'], epoch)
            writer.add_scalar('PR_AUC/val', val_metrics['pr_auc'], epoch)
            
            # Learning Rate
            writer.add_scalar('LearningRate', current_lr, epoch)
        
        if epoch == 1 or epoch % 5 == 0:
            print(f"\nEpoch {epoch}:")
            print(f"  Train - Loss={train_loss:.4f}, Acc={train_metrics['accuracy']:.4f}, "
                  f"P={train_metrics['precision']:.4f}, R={train_metrics['recall']:.4f}, F1={train_metrics['f1']:.4f}")
            print(f"          ROC-AUC={train_metrics['roc_auc']:.4f}, PR-AUC={train_metrics['pr_auc']:.4f}")
            print(f"  Val   - Loss={val_loss:.4f}, Acc={val_metrics['accuracy']:.4f}, "
                  f"P={val_metrics['precision']:.4f}, R={val_metrics['recall']:.4f}, F1={val_metrics['f1']:.4f}")
            print(f"          ROC-AUC={val_metrics['roc_auc']:.4f}, PR-AUC={val_metrics['pr_auc']:.4f}")
            print(f"  LR={current_lr:.2e}")
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = model.state_dict().copy()
        
        if early_stopping(val_loss, model):
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    if writer:
        # Log final metrics as hyperparameters
        writer.add_hparams(
            {
                'attn_model': attn_model,
                'n_qubits': config.get('max_qubits', 8),
                'n_layers': config['q_depths'][0],
                'total_params': num_params,
            },
            {
                'hparam/best_val_f1': best_val_f1,
                'hparam/final_val_acc': history['val_accuracy'][-1] if history['val_accuracy'] else 0,
                'hparam/final_val_roc_auc': history['val_roc_auc'][-1] if history['val_roc_auc'] else 0,
                'hparam/final_val_pr_auc': history['val_pr_auc'][-1] if history['val_pr_auc'] else 0,
            }
        )
        writer.close()
    
    return model, history, best_model_state, best_val_f1, num_params


def main():
    """Main training function - OPTIMIZED."""
    
    # ================================================================
    # OPTIMIZED CONFIGURATION
    # ================================================================
    config = {
        # Data
        'batch_size': 32,           # Smaller batches for quantum circuits
        
        # Model - USER REQUESTED: layers [1,3,5], qubits [4,8,16]
        'q_depths': [2],            # Default depth (will vary per experiment)
        'max_qubits': 8,            # Default qubits (will vary per experiment)
        'dropout': 0.3,             # Slightly more dropout
        
        # Training - OPTIMIZED
        'learning_rate': 0.0005,    # Lower base LR (scheduler will increase)
        'weight_decay': 1e-3,       # More regularization
        'max_epochs': 30,           # Fewer epochs (converges faster with scheduler)
        'patience': 8,              # Early stopping patience
        'validate_every': 1,        # Validate every epoch
        'grad_clip': 1.0,           # Gradient clipping
        
        # Class imbalance handling
        'use_class_weights': True,  # Handle imbalanced data
        'use_focal_loss': False,    # Alternative: Focal Loss
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        
        # Scheduler
        'use_scheduler': True,      # OneCycleLR
    }
    
    # ================================================================
    # EXPERIMENT GRID: Layer depths x Qubit counts x Attention model (HEA only)
    # ================================================================
    layer_depths = [1, 3, 5]        # USER REQUESTED: quantum circuit depths
    qubit_options = [4, 8, 16]      # USER REQUESTED: qubit counts
    
    # Train on all vulnerability types with HEA attention model only
    vulnerability_types = ['integeroverflow', 'reentrancy', 'timestamp', 'delegatecall']
    attention_models = ['HEA']
    
    base_data_dir = Path('train_data')
    results_dir = Path('training_results_1')
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    all_results = {}
    
    # Calculate total experiments
    total_experiments = len(vulnerability_types) * len(attention_models) * len(layer_depths) * len(qubit_options)
    
    print("=" * 80)
    print("QUANTUM GAT TRAINING - GRID SEARCH")
    print("=" * 80)
    print(f"Layer depths: {layer_depths}")
    print(f"Qubit options: {qubit_options}")
    print(f"Attention models: {attention_models}")
    print(f"Vulnerability types: {vulnerability_types}")
    print(f"Total experiments: {total_experiments}")
    print(f"Training: lr={config['learning_rate']}, epochs={config['max_epochs']}, "
          f"scheduler={'OneCycleLR' if config['use_scheduler'] else 'None'}")
    print("=" * 80)
    
    experiment_count = 0
    
    # Train on each vulnerability type
    for vuln_type in vulnerability_types:
        print(f"\n{'#'*80}")
        print(f"Processing vulnerability type: {vuln_type.upper()}")
        print(f"{'#'*80}")
        
        train_path = base_data_dir / vuln_type / 'train.json'
        valid_path = base_data_dir / vuln_type / 'valid.json'
        
        if not train_path.exists() or not valid_path.exists():
            print(f"Warning: Data files not found for {vuln_type}. Skipping...")
            continue
        
        # Load data (load once, no filtering)
        print(f"\nLoading {vuln_type} dataset...")
        train_data = load_vulnerability_data(str(train_path))
        val_data = load_vulnerability_data(str(valid_path))
        
        print(f"Train samples: {len(train_data)}")
        print(f"Val samples: {len(val_data)}")
        
        # ================================================================
        # COMPUTE CLASS WEIGHTS for imbalanced data
        # ================================================================
        pos_weight = None
        if config['use_class_weights']:
            pos_weight = compute_class_weights(train_data)
        
        train_loader = DataLoader(
            train_data, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_data, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        # Train all combinations: attention x layers x qubits
        vuln_results = {}
        
        for n_qubits in qubit_options:
            for n_layers in layer_depths:
                for attn_model in attention_models:
                    experiment_count += 1
                    experiment_name = f"{attn_model}_L{n_layers}_Q{n_qubits}"
                    
                    print(f"\n[{experiment_count}/{total_experiments}] "
                          f"{vuln_type} - {experiment_name}")
                    
                    try:
                        # Update config for this experiment
                        exp_config = config.copy()
                        exp_config['q_depths'] = [n_layers]
                        exp_config['max_qubits'] = n_qubits
                        
                        model_dir = results_dir / f"{vuln_type}_{experiment_name}_{timestamp}"
                        model_dir.mkdir(exist_ok=True)
                        
                        model, history, best_state, best_f1, num_params = train_model(
                            vuln_type, attn_model, train_loader, val_loader, exp_config, 
                            str(model_dir / 'logs'),
                            pos_weight=pos_weight
                        )
                        
                        # Save model and results
                        torch.save(best_state or model.state_dict(), model_dir / 'best_model.pt')
                        torch.save(model.state_dict(), model_dir / 'model.pt')
                        with open(model_dir / 'history.json', 'w') as f:
                            json.dump(history, f, indent=2)
                        with open(model_dir / 'config.json', 'w') as f:
                            json.dump(exp_config, f, indent=2)
                        
                        vuln_results[experiment_name] = {
                            'best_val_f1': best_f1,
                            'path': str(model_dir),
                            'num_params': num_params,
                            'final_train_f1': history['train_f1'][-1] if history['train_f1'] else 0,
                            'final_val_f1': history['val_f1'][-1] if history['val_f1'] else 0,
                            'final_val_accuracy': history['val_accuracy'][-1] if history['val_accuracy'] else 0,
                            'final_val_precision': history['val_precision'][-1] if history['val_precision'] else 0,
                            'final_val_recall': history['val_recall'][-1] if history['val_recall'] else 0,
                            'final_val_roc_auc': history['val_roc_auc'][-1] if history['val_roc_auc'] else 0,
                            'final_val_pr_auc': history['val_pr_auc'][-1] if history['val_pr_auc'] else 0,
                            'attn_model': attn_model,
                            'n_layers': n_layers,
                            'n_qubits': n_qubits
                        }
                        print(f"\n[OK] {experiment_name}: Best Val F1 = {best_f1:.4f}, Parameters = {num_params}")
                        
                    except Exception as e:
                        print(f"\n[ERROR] Error training {experiment_name} for {vuln_type}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
        
        all_results[vuln_type] = vuln_results
        
        # Summary for this vulnerability type
        if vuln_results:
            print(f"\n{'='*60}")
            print(f"SUMMARY - {vuln_type.upper()}")
            print(f"{'='*60}")
            
            # Group by attention model
            for attn in attention_models:
                print(f"\n  {attn}:")
                for name, res in sorted(vuln_results.items()):
                    if res['attn_model'] == attn:
                        print(f"    L{res['n_layers']}_Q{res['n_qubits']}: F1 = {res['best_val_f1']:.4f}, Params = {res.get('num_params', 'N/A')}")
            
            best_exp = max(vuln_results, key=lambda x: vuln_results[x]['best_val_f1'])
            best_res = vuln_results[best_exp]
            print(f"\n[BEST] {vuln_type}: {best_exp} (F1={best_res['best_val_f1']:.4f})")
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL TRAINING SUMMARY - GRID SEARCH RESULTS")
    print(f"{'='*80}")
    print(f"Layer depths tested: {layer_depths}")
    print(f"Qubit counts tested: {qubit_options}")
    print(f"Attention models: {attention_models}")
    print(f"{'='*80}")
    
    for vuln_type, vuln_results in all_results.items():
        if vuln_results:
            print(f"\n{vuln_type.upper()}:")
            
            # Find best for each attention model
            for attn in attention_models:
                attn_results = {k: v for k, v in vuln_results.items() if v['attn_model'] == attn}
                if attn_results:
                    best_name = max(attn_results, key=lambda x: attn_results[x]['best_val_f1'])
                    best = attn_results[best_name]
                    print(f"  {attn}: Best = L{best['n_layers']}_Q{best['n_qubits']} (F1={best['best_val_f1']:.4f})")
            
            # Overall best
            if vuln_results:
                overall_best = max(vuln_results, key=lambda x: vuln_results[x]['best_val_f1'])
                print(f"  >>> OVERALL BEST: {overall_best} (F1={vuln_results[overall_best]['best_val_f1']:.4f})")
    
    # Save overall summary
    summary_path = results_dir / f'summary_{timestamp}.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    
    # Also save a CSV for easy analysis with all metrics
    csv_path = results_dir / f'results_{timestamp}.csv'
    with open(csv_path, 'w') as f:
        f.write("vuln_type,attn_model,n_layers,n_qubits,num_params,best_val_f1,final_val_accuracy,final_val_precision,final_val_recall,final_val_f1,final_val_roc_auc,final_val_pr_auc,final_train_f1\n")
        for vuln_type, vuln_results in all_results.items():
            for exp_name, res in vuln_results.items():
                f.write(f"{vuln_type},{res['attn_model']},{res['n_layers']},{res['n_qubits']},"
                       f"{res.get('num_params', 0)},{res['best_val_f1']:.4f},"
                       f"{res.get('final_val_accuracy', 0):.4f},"
                       f"{res.get('final_val_precision', 0):.4f},{res.get('final_val_recall', 0):.4f},"
                       f"{res['final_val_f1']:.4f},{res.get('final_val_roc_auc', 0):.4f},"
                       f"{res.get('final_val_pr_auc', 0):.4f},{res['final_train_f1']:.4f}\n")
    print(f"Results CSV saved to: {csv_path}")


if __name__ == '__main__':
    main()
