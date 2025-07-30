# Astronomy ML Curriculum: From Python to Transformers

## Phase 1: Python Foundations with Astronomy Context (Exercises 1-4)

### Exercise 1: Advanced Functions for Light Curve Processing (15-30 min)
**Objective:** Master args/kwargs, decorators, and higher-order functions using photometric data processing.

**Starter Code:**
```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, Any, Optional
from functools import wraps
import time

# Sample light curve data (time, flux, flux_err, band)
def generate_sample_lightcurve(n_points=50, bands=['g', 'r', 'i']):
    """Generate synthetic multi-band light curve data"""
    np.random.seed(42)
    times = np.sort(np.random.uniform(0, 100, n_points))
    data = []
    for band in bands:
        flux = 20 + 2 * np.sin(0.1 * times) + np.random.normal(0, 0.5, n_points)
        flux_err = 0.1 + 0.05 * np.random.random(n_points)
        for t, f, fe in zip(times, flux, flux_err):
            data.append({'mjd': t, 'flux': f, 'flux_err': fe, 'band': band})
    return data

# TODO: Implement these functions
def timing_decorator(func):
    """Decorator to time function execution"""
    pass

def filter_lightcurve(data: list, **filters) -> list:
    """Filter light curve data by arbitrary criteria using **kwargs
    Example: filter_lightcurve(data, band='g', mjd_min=10, flux_min=19)
    """
    pass

def apply_magnitude_conversion(data: list, 
                             conversion_func: Callable[[float], float],
                             *args, **kwargs) -> list:
    """Apply conversion function to flux values using higher-order functions"""
    pass

def flux_to_magnitude(flux: float, zeropoint: float = 25.0) -> float:
    """Convert flux to magnitude"""
    return -2.5 * np.log10(flux) + zeropoint
```

**Expected Outputs:**
- `timing_decorator` should print execution time
- `filter_lightcurve` should handle arbitrary keyword filters
- `apply_magnitude_conversion` should transform flux→magnitude for all data points
- Demonstrate usage with sample data

**Extensions:**
1. Add a `cache_results` decorator for expensive computations
2. Create a `band_processor` higher-order function that applies different functions per band

---

### Exercise 2: Light Curve Classes and Magic Methods (30-45 min)
**Objective:** Build OOP foundations with `__init__`, magic methods, and inheritance.

**Starter Code:**
```python
import pandas as pd
import numpy as np
from typing import List, Optional, Union

class LightCurve:
    """Base class for astronomical light curves"""
    
    def __init__(self, data: List[Dict], object_id: str = "unknown"):
        # TODO: Initialize instance variables
        # Store data as pandas DataFrame with columns: mjd, flux, flux_err, band
        pass
    
    def __len__(self):
        """Return number of observations"""
        pass
    
    def __getitem__(self, key):
        """Enable indexing: lc[0] returns first observation, lc['g'] returns g-band data"""
        pass
    
    def __str__(self):
        """Human readable representation"""
        pass
    
    def __repr__(self):
        """Developer representation"""
        pass
    
    @property
    def bands(self) -> List[str]:
        """Return unique photometric bands"""
        pass
    
    @property
    def duration(self) -> float:
        """Return time span of observations"""
        pass

class TransientLightCurve(LightCurve):
    """Specialized class for transient events"""
    
    def __init__(self, data: List[Dict], object_id: str, 
                 transient_type: Optional[str] = None,
                 peak_mjd: Optional[float] = None):
        # TODO: Use super().__init__() and add transient-specific attributes
        pass
    
    def phase_fold(self, period: float) -> 'TransientLightCurve':
        """Fold light curve by given period"""
        pass
    
    def get_peak_magnitude(self, band: str = 'r') -> float:
        """Find peak (minimum) magnitude in specified band"""
        pass
```

**Expected Outputs:**
- Create `LightCurve` and `TransientLightCurve` objects
- Demonstrate all magic methods work correctly
- Show inheritance with `super().__init__()`
- Use properties to access derived information

**Extensions:**
1. Add `__add__` method to combine light curves
2. Implement `__iter__` to make light curve iterable by observation

---

### Exercise 3: Advanced Class Design for Astronomy Surveys (30-45 min)
**Objective:** Master static/class methods, advanced inheritance patterns.

**Starter Code:**
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Dict, List
import requests
import io

@dataclass
class SurveyConfig:
    """Configuration for astronomical surveys"""
    name: str
    bands: List[str]
    zeropoints: Dict[str, float]
    cadence_days: float

class SurveyProcessor(ABC):
    """Abstract base class for survey data processing"""
    
    # Class variable shared by all surveys
    SUPPORTED_FORMATS: ClassVar[List[str]] = ['csv', 'fits', 'json']
    
    def __init__(self, config: SurveyConfig):
        self.config = config
        self._processed_count = 0
    
    @classmethod
    def from_survey_name(cls, survey_name: str):
        """Factory method to create processor from survey name"""
        # TODO: Implement factory pattern
        pass
    
    @staticmethod
    def validate_photometry(flux: float, flux_err: float) -> bool:
        """Validate photometric measurement (no self needed)"""
        # TODO: Check for reasonable flux/error values
        pass
    
    @abstractmethod
    def process_lightcurve(self, raw_data: Dict) -> LightCurve:
        """Process raw survey data to LightCurve object"""
        pass
    
    @property
    def processed_count(self) -> int:
        return self._processed_count

class ZTFProcessor(SurveyProcessor):
    """ZTF survey processor implementation"""
    
    def __init__(self):
        config = SurveyConfig(
            name="ZTF",
            bands=['g', 'r', 'i'],
            zeropoints={'g': 26.325, 'r': 26.275, 'i': 25.660},
            cadence_days=3.0
        )
        super().__init__(config)
    
    def process_lightcurve(self, raw_data: Dict) -> TransientLightCurve:
        # TODO: Convert ZTF format to TransientLightCurve
        pass

class LSSTProcessor(SurveyProcessor):
    """LSST survey processor (future implementation)"""
    
    def __init__(self):
        # TODO: Implement LSST-specific configuration
        pass
    
    def process_lightcurve(self, raw_data: Dict) -> TransientLightCurve:
        # TODO: Implement LSST processing
        pass
```

**Expected Outputs:**
- Implement factory method to create appropriate processor
- Show static method usage for validation
- Demonstrate abstract base class enforcement
- Create working ZTF and LSST processors

**Extensions:**
1. Add `@property` decorators for computed survey statistics
2. Implement context manager (`__enter__`, `__exit__`) for batch processing

---

### Exercise 4: Design Patterns for Astronomy Pipeline (45-60 min)
**Objective:** Combine all Python concepts into a realistic astronomy data pipeline.

**Starter Code:**
```python
from contextlib import contextmanager
from typing import Iterator, Protocol
import logging

# Protocol for type hinting
class Classifier(Protocol):
    def predict(self, lightcurve: LightCurve) -> str:
        ...

class AstronomyPipeline:
    """Complete pipeline for transient classification"""
    
    def __init__(self, processor: SurveyProcessor, classifier: Classifier):
        self.processor = processor
        self.classifier = classifier
        self.logger = self._setup_logging()
    
    @staticmethod
    def _setup_logging():
        """Setup logging configuration"""
        pass
    
    @contextmanager
    def batch_processing_session(self) -> Iterator[None]:
        """Context manager for batch processing with cleanup"""
        pass
    
    def __call__(self, raw_observations: List[Dict]) -> List[str]:
        """Make pipeline callable for easy usage"""
        pass
    
    @timing_decorator  # Use decorator from Exercise 1
    def process_and_classify(self, raw_data: Dict) -> str:
        """Complete processing pipeline"""
        pass

# TODO: Implement a simple rule-based classifier
class SimpleTransientClassifier:
    """Rule-based classifier using light curve features"""
    
    def __init__(self):
        pass
    
    def predict(self, lightcurve: TransientLightCurve) -> str:
        """Classify based on simple rules (rise time, peak magnitude, etc.)"""
        pass
```

**Expected Outputs:**
- Working pipeline that processes raw data → light curves → classifications
- Demonstrate context manager usage
- Show Protocol usage for type hints
- Integration of all previous concepts

**Extensions:**
1. Add async processing capabilities
2. Implement observer pattern for pipeline event notifications

---

## Phase 2: PyTorch Foundations for Astronomy (Exercises 5-8)

### Exercise 5: Custom Dataset for Light Curve Data (30-45 min)
**Objective:** Create PyTorch `Dataset` and `DataLoader` for irregular time-series astronomy data.

**Starter Code:**
```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import LabelEncoder

class LightCurveDataset(Dataset):
    """PyTorch Dataset for multi-band light curve data"""
    
    def __init__(self, 
                 lightcurves: List[TransientLightCurve],
                 labels: List[str],
                 max_sequence_length: int = 100,
                 transform: Optional[callable] = None):
        # TODO: Initialize dataset
        # Handle variable-length sequences, encode labels
        pass
    
    def __len__(self) -> int:
        pass
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (features, label) tuple
        Features shape: (max_seq_len, n_features)
        Features include: [mjd, flux, flux_err, band_encoded]
        """
        pass
    
    def _pad_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Pad or truncate sequence to max_sequence_length"""
        pass
    
    def _encode_bands(self, bands: List[str]) -> np.ndarray:
        """One-hot encode photometric bands"""
        pass

# Data augmentation for light curves
class LightCurveAugmentation:
    """Data augmentation techniques for light curves"""
    
    def __init__(self, noise_level: float = 0.01, time_shift_max: float = 5.0):
        self.noise_level = noise_level
        self.time_shift_max = time_shift_max
    
    def __call__(self, lightcurve_data: np.ndarray) -> np.ndarray:
        """Apply random augmentations"""
        # TODO: Add noise, time shifts, missing data simulation
        pass

# TODO: Generate synthetic dataset
def create_synthetic_transient_dataset(n_samples: int = 1000) -> Tuple[List[TransientLightCurve], List[str]]:
    """Create synthetic dataset with different transient types"""
    # SNIa, SNII, CV, AGN, TDE patterns
    pass
```

**Expected Outputs:**
- Working `Dataset` that handles variable-length sequences
- Proper padding/truncation for batch processing
- Data augmentation pipeline
- Test with `DataLoader` showing batched data

**Extensions:**
1. Add sophisticated augmentations (photometric redshift simulation)
2. Implement stratified sampling for imbalanced classes

---

### Exercise 6: Basic Neural Networks for Photometry (30-45 min)
**Objective:** Build custom `nn.Module` networks, understand forward passes and training loops.

**Starter Code:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class SimplePhotometryMLP(nn.Module):
    """Basic MLP for light curve classification"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, dropout: float = 0.2):
        super().__init__()
        # TODO: Build sequential layers with dropout
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass
        Args:
            x: (batch_size, sequence_length, features)
        Returns:
            logits: (batch_size, num_classes)
        """
        # TODO: Handle sequence data (maybe global average pooling?)
        pass

class LightCurveLSTM(nn.Module):
    """LSTM-based classifier for sequential light curve data"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 num_classes: int, dropout: float = 0.2):
        super().__init__()
        # TODO: Implement LSTM + classifier head
        pass
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Forward pass with proper sequence length handling"""
        # TODO: Use pack_padded_sequence for efficiency
        pass

# Training utilities
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance (rare transients)"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # TODO: Implement focal loss formula
        pass

def train_epoch(model: nn.Module, dataloader: DataLoader, 
                optimizer: torch.optim.Optimizer, criterion: nn.Module,
                device: str = 'cpu') -> float:
    """Training loop for one epoch"""
    # TODO: Implement training loop with proper gradient handling
    pass

def evaluate_model(model: nn.Module, dataloader: DataLoader, 
                  criterion: nn.Module, device: str = 'cpu') -> Tuple[float, float]:
    """Evaluation loop returning loss and accuracy"""
    # TODO: Implement evaluation with no_grad()
    pass
```

**Expected Outputs:**
- Working MLP and LSTM models
- Focal loss implementation for imbalanced data
- Complete training/evaluation loops
- Demonstrate overfitting on small synthetic dataset

**Extensions:**
1. Add learning rate scheduling
2. Implement early stopping based on validation loss

---

### Exercise 7: Time2Vec Embeddings for Temporal Data (45-60 min)
**Objective:** Implement Time2Vec positional encoding for irregular time-series data.

**Starter Code:**
```python
import torch
import torch.nn as nn
import math

class Time2Vec(nn.Module):
    """Time2Vec: Learning a Vector Representation of Time
    Paper: https://arxiv.org/abs/1907.05321
    """
    
    def __init__(self, input_dim: int = 1, embed_dim: int = 64):
        super().__init__()
        # TODO: Implement Time2Vec embedding
        # One linear transformation + multiple periodic transformations
        pass
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: (batch_size, seq_len, 1) - MJD timestamps
        Returns:
            embeddings: (batch_size, seq_len, embed_dim)
        """
        pass

class PositionalTime2Vec(nn.Module):
    """Combine relative position and absolute time information"""
    
    def __init__(self, time_embed_dim: int = 64, pos_embed_dim: int = 64):
        super().__init__()
        self.time2vec = Time2Vec(embed_dim=time_embed_dim)
        # TODO: Add learnable positional embeddings
        pass
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Combine time and position embeddings"""
        pass

class Time2VecTransformerEncoder(nn.Module):
    """Transformer encoder with Time2Vec temporal embeddings"""
    
    def __init__(self, 
                 input_dim: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 time_embed_dim: int = 64,
                 num_classes: int = 5,
                 max_seq_len: int = 100):
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model - time_embed_dim)
        
        # Time2Vec embeddings
        self.time_embedding = Time2Vec(embed_dim=time_embed_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def create_padding_mask(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Create attention mask for padded sequences"""
        # TODO: Create boolean mask for padding tokens
        pass
    
    def forward(self, features: torch.Tensor, times: torch.Tensor, 
                lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, seq_len, feature_dim) - flux, flux_err, band_encoding
            times: (batch, seq_len, 1) - MJD timestamps  
            lengths: (batch,) - actual sequence lengths
        """
        # TODO: Implement full forward pass
        # 1. Project input features
        # 2. Get time embeddings
        # 3. Concatenate features + time embeddings
        # 4. Apply transformer with padding mask
        # 5. Global average pooling over sequence
        # 6. Classification
        pass
```

**Expected Outputs:**
- Working Time2Vec embedding layer
- Complete transformer architecture with temporal embeddings
- Proper handling of variable-length sequences
- Test on synthetic periodic/aperiodic signals

**Extensions:**
1. Add learnable band embeddings in addition to time
2. Implement causal attention for real-time classification

---

### Exercise 8: Advanced Training with Class Imbalance (45-60 min)
**Objective:** Handle rare astronomical events, implement advanced training techniques.

**Starter Code:**
```python
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class BalancedBatchSampler:
    """Custom sampler to ensure balanced batches for rare classes"""
    
    def __init__(self, labels: List[int], batch_size: int = 32):
        self.labels = labels
        self.batch_size = batch_size
        # TODO: Implement balanced sampling strategy
        pass
    
    def __iter__(self):
        # TODO: Yield balanced batches
        pass

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss: float) -> bool:
        """Returns True if training should stop"""
        # TODO: Implement early stopping logic
        pass

class MetricsTracker:
    """Track training metrics for astronomy-specific evaluation"""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, loss: float):
        # TODO: Accumulate predictions and targets
        pass
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute comprehensive metrics including rare event recall"""
        # TODO: Return precision, recall, F1 for each class
        # Focus on rare event detection (TDE, rare SNe)
        pass
    
    def plot_confusion_matrix(self) -> plt.Figure:
        """Plot confusion matrix with class names"""
        pass

def train_with_imbalance_handling(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    device: str = 'cpu'
):
    """Complete training pipeline with imbalance handling"""
    
    # TODO: Implement comprehensive training loop with:
    # 1. Focal loss or weighted cross-entropy
    # 2. Learning rate scheduling
    # 3. Early stopping
    # 4. Metrics tracking
    # 5. Model checkpointing
    
    # Setup
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    early_stopping = EarlyStopping(patience=15)
    
    # Class weights for imbalanced data
    # TODO: Compute and apply class weights
    
    training_history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'learning_rates': []
    }
    
    for epoch in range(num_epochs):
        # TODO: Implement training epoch
        # TODO: Implement validation epoch
        # TODO: Apply scheduler and early stopping
        pass
    
    return model, training_history
```

**Expected Outputs:**
- Balanced batch sampling for rare events
- Working early stopping implementation
- Comprehensive metrics including per-class performance
- Training history visualization

**Extensions:**
1. Implement mixup augmentation for minority classes
2. Add uncertainty estimation with Monte Carlo dropout

---

## Phase 3: Weights & Biases Integration (Exercises 9-10)

### Exercise 9: Experiment Tracking and Hyperparameter Sweeps (45-60 min)
**Objective:** Integrate W&B for experiment tracking, hyperparameter optimization.

**Starter Code:**
```python
import wandb
from itertools import product
import yaml

class WandBExperimentTracker:
    """Wrapper for Weights & Biases experiment tracking"""
    
    def __init__(self, project_name: str = "astro-transient-classification"):
        self.project_name = project_name
        self.run = None
    
    def start_experiment(self, config: Dict[str, Any], experiment_name: str = None):
        """Initialize W&B run with configuration"""
        # TODO: Initialize wandb run
        # Log hyperparameters, model architecture info
        pass
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log training metrics"""
        pass
    
    def log_model_checkpoint(self, model: nn.Module, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint as W&B artifact"""
        pass
    
    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           class_names: List[str]):
        """Log confusion matrix visualization"""
        pass
    
    def log_light_curve_predictions(self, lightcurves: List[TransientLightCurve], 
                                  predictions: np.ndarray, true_labels: np.ndarray,
                                  n_examples: int = 10):
        """Log example light curves with predictions"""
        # TODO: Create matplotlib figures and log to W&B
        pass
    
    def finish_experiment(self):
        """Clean up W&B run"""
        if self.run:
            wandb.finish()

# Hyperparameter sweep configuration
def create_sweep_config() -> Dict[str, Any]:
    """Define hyperparameter sweep for transformer model"""
    sweep_config = {
        'method': 'bayes',  # or 'grid', 'random'
        'metric': {
            'name': 'val_f1_rare_events',  # Focus on rare event detection
            'goal': 'maximize'
        },
        'parameters': {
            # TODO: Define hyperparameter ranges
            'd_model': {'values': [128, 256, 512]},
            'nhead': {'values': [4, 8, 16]},
            'num_layers': {'values': [3, 6, 9]},
            'learning_rate': {'distribution': 'log_uniform_values', 
                            'min': 1e-5, 'max': 1e-2},
            'time_embed_dim': {'values': [32, 64, 128]},
            'dropout': {'distribution': 'uniform', 'min': 0.1, 'max': 0.5},
            'batch_size': {'values': [16, 32, 64]},
            'focal_loss_gamma': {'distribution': 'uniform', 'min': 1.0, 'max': 3.0}
        }
    }
    return sweep_config

def train_with_wandb(config: Dict[str, Any] = None):
    """Training function for W&B sweep"""
    
    # Initialize W&B
    wandb.init(config=config)
    config = wandb.config
    
    # TODO: Create model with sweep parameters
    model = Time2VecTransformerEncoder(
        input_dim=4,  # mjd, flux, flux_err, band_encoding
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        time_embed_dim=config.time_embed_dim,
        num_classes=5  # SNIa, SNII, CV, AGN, TDE
    )
    
    # TODO: Create datasets and dataloaders
    # TODO: Initialize training components
    
    # Training loop with W&B logging
    for epoch in range(100):
        # TODO: Training epoch
        # Log metrics: wandb.log({'train_loss': loss, 'epoch': epoch})
        
        # TODO: Validation epoch
        # TODO: Log additional metrics and visualizations
        
        # Early stopping check
        pass
    
    # Final model evaluation and artifact logging
    # TODO: Evaluate on test set
    # TODO: Log final model as artifact
    
    wandb.finish()
```

**Expected Outputs:**
- Complete W&B integration with metric logging
- Hyperparameter sweep configuration
- Model checkpointing and artifact management
- Rich visualizations (confusion matrices, light curve examples)

**Extensions:**
1. Add custom W&B charts for astronomy-specific metrics
2. Implement automatic model deployment based on performance

---

### Exercise 10: Bayesian Uncertainty and Model Comparison (60+ min)
**Objective:** Add uncertainty estimation, compare multiple architectures, advanced W&B features.

**Starter Code:**
```python
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class BayesianTransformerEncoder(Time2VecTransformerEncoder):
    """Transformer with Monte Carlo Dropout for uncertainty estimation"""
    
    def __init__(self, *args, mc_dropout_rate: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.mc_dropout_rate = mc_dropout_rate
        
        # Replace standard dropout with MC dropout
        self._replace_dropout_layers()
    
    def _replace_dropout_layers(self):
        """Replace standard dropout with MC dropout"""
        # TODO: Replace dropout layers to stay active during inference
        pass
    
    def forward_with_uncertainty(self, *args, n_samples: int = 100, **kwargs):
        """Forward pass with uncertainty estimation"""
        self.train()  # Keep dropout active
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(*args, **kwargs)
                predictions.append(F.softmax(pred, dim=-1))
        
        predictions = torch.stack(predictions)  # (n_samples, batch_size, n_classes)
        
        # Compute statistics
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = std_pred.mean(dim=-1)
        
        # Aleatoric uncertainty (data uncertainty) - approximation
        aleatoric_uncertainty = (mean_pred * (1 - mean_pred)).sum(dim=-1)
        
        return {
            'predictions': mean_pred,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'prediction_samples': predictions
        }

class ModelComparison:
    """Compare multiple model architectures with W&B"""
    
    def __init__(self, project_name: str = "astro-model-comparison"):
        self.project_name = project_name
        self.results = {}
    
    def train_and_evaluate_model(self, model_name: str, model_class: type, 
                                model_config: Dict[str, Any],
                                train_loader: DataLoader, val_loader: DataLoader,
                                test_loader: DataLoader):
        """Train and evaluate a single model"""
        
        # Start W&B run
        with wandb.init(project=self.project_name, name=model_name, 
                       config=model_config) as run:
            
            # TODO: Initialize and train model
            # TODO: Evaluate on test set
            # TODO: Compute uncertainty metrics
            # TODO: Log comprehensive results
            
            # Store results for comparison
            self.results[model_name] = {
                'test_accuracy': 0.0,  # TODO: Fill with actual results
                'rare_event_f1': 0.0,
                'mean_uncertainty': 0.0,
                'calibration_error': 0.0  # How well-calibrated are uncertainties?
            }
    
    def compare_models(self):
        """Create comprehensive model comparison report"""
        # TODO: Generate comparison tables and plots
        # TODO: Create W&B report with model comparison
        pass

class UncertaintyCalibration:
    """Tools for evaluating uncertainty calibration"""
    
    @staticmethod
    def reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray, 
                          uncertainty: np.ndarray, n_bins: int = 10):
        """Create reliability diagram for uncertainty calibration"""
        # TODO: Bin predictions by confidence/uncertainty
        # TODO: Plot expected vs observed accuracy
        pass
    
    @staticmethod
    def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, 
                                 n_bins: int = 10) -> float:
        """Compute Expected Calibration Error (ECE)"""
        # TODO: Implement ECE calculation
        pass

# Advanced experiment configuration
def create_advanced_sweep_config():
    """Advanced sweep including architecture search"""
    return {
        'method': 'bayes',
        'metric': {'name': 'calibrated_rare_event_score', 'goal': 'maximize'},
        'parameters': {
            # Architecture choices
            'architecture': {
                'values': ['transformer', 'lstm', 'cnn_lstm', 'resnet_transformer']
            },
            
            # Transformer-specific
            'd_model': {'distribution': 'int_uniform', 'min': 64, 'max': 512},
            'nhead': {'values': [2, 4, 8, 16]},
            'num_layers': {'distribution': 'int_uniform', 'min': 2, 'max': 12},
            
            # Time encoding
            'time_encoding': {'values': ['time2vec', 'sinusoidal', 'learned']},
            'time_embed_dim': {'distribution': 'int_uniform', 'min': 16, 'max': 128},
            
            # Training
            'optimizer': {'values': ['adam', 'adamw', 'sgd']},
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-2},
            'weight_decay': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-2},
            'scheduler': {'values': ['plateau', 'cosine', 'exponential', 'none']},
            
            # Regularization
            'dropout': {'distribution': 'uniform', 'min': 0.05, 'max': 0.4},
            'mc_dropout': {'values': [True, False]},
            'label_smoothing': {'distribution': 'uniform', 'min': 0.0, 'max': 0.2},
            
            # Loss function
            'loss_type': {'values': ['focal', 'weighted_ce', 'class_balanced']},
            'focal_gamma': {'distribution': 'uniform', 'min': 0.5, 'max': 3.0},
            'focal_alpha': {'distribution': 'uniform', 'min': 0.25, 'max': 1.0},
            
            # Data augmentation
            'augmentation_prob': {'distribution': 'uniform', 'min': 0.0, 'max': 0.8},
            'noise_augmentation': {'distribution': 'uniform', 'min': 0.01, 'max': 0.1},
            'time_shift_augmentation': {'distribution': 'uniform', 'min': 0.0, 'max': 10.0}
        }
    }

# Example usage and final integration
def main_experiment_pipeline():
    """Complete experiment pipeline demonstration"""
    
    # 1. Data preparation
    print("Preparing synthetic transient dataset...")
    lightcurves, labels = create_synthetic_transient_dataset(n_samples=5000)
    
    # Split data
    from sklearn.model_selection import train_test_split
    train_lcs, test_lcs, train_labels, test_labels = train_test_split(
        lightcurves, labels, test_size=0.2, stratify=labels, random_state=42
    )
    train_lcs, val_lcs, train_labels, val_labels = train_test_split(
        train_lcs, train_labels, test_size=0.2, stratify=train_labels, random_state=42
    )
    
    # Create datasets
    train_dataset = LightCurveDataset(train_lcs, train_labels, transform=LightCurveAugmentation())
    val_dataset = LightCurveDataset(val_lcs, val_labels)
    test_dataset = LightCurveDataset(test_lcs, test_labels)
    
    # 2. Model comparison
    print("Starting model comparison...")
    comparison = ModelComparison()
    
    # Compare different architectures
    models_to_compare = {
        'baseline_lstm': {
            'class': LightCurveLSTM,
            'config': {'input_size': 4, 'hidden_size': 128, 'num_layers': 2, 'num_classes': 5}
        },
        'time2vec_transformer': {
            'class': Time2VecTransformerEncoder,
            'config': {'input_dim': 4, 'd_model': 256, 'nhead': 8, 'num_layers': 6, 'num_classes': 5}
        },
        'bayesian_transformer': {
            'class': BayesianTransformerEncoder,
            'config': {'input_dim': 4, 'd_model': 256, 'nhead': 8, 'num_layers': 6, 'num_classes': 5, 'mc_dropout_rate': 0.1}
        }
    }
    
    for model_name, model_info in models_to_compare.items():
        comparison.train_and_evaluate_model(
            model_name, model_info['class'], model_info['config'],
            DataLoader(train_dataset, batch_size=32, shuffle=True),
            DataLoader(val_dataset, batch_size=32),
            DataLoader(test_dataset, batch_size=32)
        )
    
    # 3. Hyperparameter optimization
    print("Starting hyperparameter sweep...")
    sweep_config = create_advanced_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project="astro-transient-optimization")
    
    # Run sweep
    wandb.agent(sweep_id, train_with_wandb, count=50)  # Run 50 experiments
    
    # 4. Final model analysis
    print("Analyzing best model...")
    # TODO: Load best model from sweep
    # TODO: Perform detailed uncertainty analysis
    # TODO: Generate final report
    
    return comparison.results

if __name__ == "__main__":
    # Run complete pipeline
    results = main_experiment_pipeline()
    print("Experiment pipeline completed!")
    print("Results:", results)
```

**Expected Outputs:**
- Working Bayesian uncertainty estimation
- Model architecture comparison framework
- Advanced hyperparameter sweep with multiple objectives
- Uncertainty calibration evaluation
- Complete end-to-end pipeline

**Extensions:**
1. Add ensemble methods for improved uncertainty
2. Implement active learning for optimal data labeling
3. Add real-time inference pipeline for survey data

---

## Phase 4: Real-World Application (Final Project)

### Final Project: End-to-End ZTF Transient Classification System
**Objective:** Build production-ready system for real ZTF data classification with uncertainty quantification.

**Project Components:**

#### 1. Data Pipeline (2-3 hours)
```python
class ZTFDataPipeline:
    """Production pipeline for ZTF forced photometry data"""
    
    def __init__(self, data_source: str = "lasair"):
        # TODO: Implement data fetching from Lasair/ANTARES
        # TODO: Handle real ZTF photometry format
        # TODO: Implement quality cuts and preprocessing
        pass
    
    def fetch_recent_transients(self, lookback_days: int = 7) -> List[TransientLightCurve]:
        """Fetch recent ZTF detections for classification"""
        pass
    
    def preprocess_for_inference(self, lightcurve: TransientLightCurve) -> torch.Tensor:
        """Standardize data for model inference"""
        pass
```

#### 2. Production Model (2-3 hours)
```python
class ProductionTransientClassifier:
    """Production-ready classifier with uncertainty quantification"""
    
    def __init__(self, model_path: str, config_path: str):
        # TODO: Load trained model from W&B artifacts
        # TODO: Load preprocessing configuration
        # TODO: Set up inference optimizations (TorchScript, ONNX)
        pass
    
    def classify_with_uncertainty(self, lightcurves: List[TransientLightCurve]):
        """Classify transients with calibrated uncertainties"""
        # TODO: Batch inference
        # TODO: Uncertainty quantification
        # TODO: Confidence calibration
        pass
    
    def explain_prediction(self, lightcurve: TransientLightCurve, prediction: Dict):
        """Generate explainable predictions using attention weights"""
        pass
```

#### 3. Monitoring and Alerting (1-2 hours)
```python
class TransientAlertSystem:
    """Real-time alerting for high-confidence rare events"""
    
    def __init__(self, alert_thresholds: Dict[str, float]):
        # TODO: Set up alert criteria (TDE > 0.8 confidence, etc.)
        pass
    
    def process_classifications(self, results: List[Dict]):
        """Process batch classifications and trigger alerts"""
        # TODO: Filter high-confidence rare events
        # TODO: Send alerts via email/Slack/etc.
        # TODO: Log to W&B for monitoring
        pass
```

#### 4. Evaluation on Real Data (1-2 hours)
- Download labeled ZTF transients from TNS (Transient Name Server)
- Evaluate model performance on real data
- Compare with existing classification systems
- Analyze failure cases and model limitations

**Final Deliverables:**
1. Complete working system with uncertainty quantification
2. Performance analysis on real ZTF data
3. W&B dashboard showing model performance over time
4. Documentation and deployment guide
5. Analysis of most challenging classification cases

**Success Criteria:**
- >90% accuracy on main transient classes (SNIa, SNII, CV, AGN)
- >70% recall on rare events (TDE) with reasonable precision
- Well-calibrated uncertainty estimates (ECE < 0.1)
- <1 second inference time per light curve
- Complete experiment tracking and reproducibility

---

## Learning Path Summary

**Phase 1 (4-6 hours):** Master Python OOP, decorators, design patterns with astronomy context
**Phase 2 (6-8 hours):** Build PyTorch skills from basic networks to advanced transformers
**Phase 3 (4-6 hours):** Implement comprehensive experiment tracking and Bayesian methods  
**Phase 4 (6-10 hours):** Deploy production system with real data

**Total Time Investment:** 20-30 hours for complete mastery

**Key Skills Acquired:**
- Advanced Python: OOP, decorators, context managers, design patterns
- PyTorch: Custom datasets, architectures, training loops, advanced losses
- Time-series ML: Time2Vec, transformers, uncertainty quantification
- MLOps: W&B tracking, hyperparameter optimization, model deployment
- Astronomy domain: Light curve processing, transient classification, survey data

**Extensions for Continued Learning:**
1. Multi-modal classification (spectra + photometry)
2. Federated learning across surveys
3. Graph neural networks for spatial-temporal correlations
4. Causal inference for physical parameter estimation
5. Integration with follow-up optimization systems

This curriculum provides a complete pathway from Python foundations to state-of-the-art astronomy ML, with each exercise building naturally toward the goal of classifying astronomical transients with modern deep learning techniques.
