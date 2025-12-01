"""
Transformer-based Deep Knowledge Tracing (DKT) Model

This script implements a transformer-based DKT model that:
1. Replaces the LSTM architecture with a Transformer Encoder
2. Uses positional encoding for sequence information
3. Implements multi-head self-attention for better pattern recognition
4. Maintains compatibility with the original DKT model interface
5. Preserves all feature processing (time, difficulty, streak, etc.)
"""
import os
import re
import math
import json
import time
import glob
import copy
import torch
import random
import pickle
import numpy as np
import pandas as pd
import platform
import datetime
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.tensorboard import SummaryWriter

# -----------------------------------------------------------------------------
# Difficulty features
# -----------------------------------------------------------------------------
try:
    from difficulty_features import build_difficulty_dictionary
except ImportError:
    print("Warning: difficulty_features module not found. Using synthetic difficulty scores.")

    def build_difficulty_dictionary(metadata_path=None):
        """Create a synthetic difficulty dictionary."""
        print("Creating synthetic difficulty dictionary...")
        difficulty_dict = {}
        for i in range(1, 10000):
            difficulty_dict[f"q{i}"] = random.random()
        return difficulty_dict

# -----------------------------------------------------------------------------
# Device selection
# -----------------------------------------------------------------------------
if hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal) device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
class TransformerConfig:
    # Data parameters
    max_sequences: int = 40000
    max_seq_len: int = 400
    batch_size: int = 16

    # Model parameters
    hidden_size: int = 96
    concept_embedding_dim: int = 96
    num_layers: int = 3
    dropout: float = 0.2
    bidirectional: bool = True
    time_embed_dim: int = 16
    use_attention: bool = True
    use_gated_skip: bool = True

    # Transformer specific parameters
    nhead: int = 4
    dim_feedforward: int = 512
    noise_std: float = 0.05

    # Training parameters
    num_epochs: int = 70
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    use_mixed_precision: bool = True
    use_cosine_scheduler: bool = True
    warmup_epochs: int = 7
    cosine_restart_epochs: int = 5
    cosine_t_mult: int = 2
    grad_accum_steps: int = 1

    # Cross-validation parameters
    k_folds: int = 5

    # Evaluation parameters
    eval_interval: int = 200

    # Logging parameters
    log_interval: int = 5
    tensorboard_dir: str = "tensorboard_logs"
    weights_dir: str = "weights"

    # Regularization parameters
    l2_reg: float = 5e-5
    early_stopping: bool = True
    patience: int = 7
    label_smoothing: float = 0.1

    # Data separation parameters
    test_size: float = 0.2
    val_size: float = 0.1
    random_seed = None

    # Feature parameters
    use_time_features: bool = True
    use_difficulty: bool = True
    difficulty_embed_dim: int = 8
    min_concept_count: int = 5

    # Time feature parameters
    normalize_time: bool = True

    # Data augmentation parameters
    use_sequence_cropping: bool = False
    crop_ratio: float = 0.1
    add_feature_noise: bool = True
    noise_level: float = 0.08
    use_cutmix: bool = True
    cutmix_prob: float = 0.3

    # Advanced features
    use_streak_features: bool = True
    use_time_since_last: bool = True

    # Knowledge distillation parameters
    use_knowledge_distillation: bool = True
    distillation_temp: float = 2.0
    distillation_alpha: float = 0.2

    # SWA parameters
    use_swa: bool = True
    swa_start: int = 10
    swa_lr: float = 5e-5
    swa_anneal_epochs: int = 5

    # Optional metadata path for difficulty construction
    metadata_path: str | None = None


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
class SmoothBCEwLogits(nn.Module):
    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        # pred and target are probabilities (model returns sigmoid already)
        if len(target.shape) > 1:
            seq_len_factor = min(1.0, target.size(1) / 100)
            adaptive_smoothing = self.smoothing * (0.8 + 0.4 * seq_len_factor)
        else:
            adaptive_smoothing = self.smoothing

        smooth_target = target * (1 - adaptive_smoothing) + 0.5 * adaptive_smoothing
        loss = F.binary_cross_entropy(pred, smooth_target)
        return loss


# -----------------------------------------------------------------------------
# Positional Encoding
# -----------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_seq_len: int = 400, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class TransformerDKTModel(nn.Module):
    """
    Transformer-based DKT model.
    """

    def __init__(
        self,
        num_concepts: int,
        hidden_size: int = 96,
        time_feature_dim: int = 16,
        difficulty_feature_dim: int = 8,
        num_layers: int = 3,
        dropout: float = 0.2,
        use_time_features: bool = False,
        use_difficulty: bool = False,
        bidirectional: bool = True,
        normalize_time: bool = True,
        use_attention: bool = False,
        use_gated_skip: bool = False,
        use_streak_features: bool = False,
        use_time_since_last: bool = False,
        nhead: int = 4,
        dim_feedforward: int = 512,
        noise_std: float = 0.05,
        pretrained_embeddings: torch.Tensor | None = None,
    ):
        super().__init__()

        self.num_concepts = num_concepts
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_time_features = use_time_features
        self.use_difficulty = use_difficulty
        self.time_feature_dim = time_feature_dim
        self.difficulty_feature_dim = difficulty_feature_dim
        self.bidirectional = bidirectional
        self.normalize_time = normalize_time
        self.use_attention = use_attention
        self.use_gated_skip = use_gated_skip
        self.use_streak_features = use_streak_features
        self.use_time_since_last = use_time_since_last
        self.nhead = nhead

        # Embeddings
        if pretrained_embeddings is not None:
            print(f"Using pretrained embeddings with shape: {pretrained_embeddings.shape}")
            self.concept_embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=False, padding_idx=0
            )
        else:
            self.concept_embedding = nn.Embedding(num_concepts + 1, hidden_size)

        self.feature_dropout = nn.Dropout2d(dropout)

        # Build input feature space
        input_size = hidden_size + 1  # concept embedding + correctness

        if use_time_features:
            if normalize_time:
                self.time_projection = nn.Sequential(nn.Linear(1, time_feature_dim), nn.GELU())
            else:
                self.time_embedding = nn.Embedding(3, time_feature_dim)
            self.time_dropout = nn.Dropout(dropout)
            input_size += time_feature_dim

        if use_difficulty:
            self.difficulty_projection = nn.Sequential(nn.Linear(1, difficulty_feature_dim), nn.GELU())
            self.difficulty_dropout = nn.Dropout(dropout)
            input_size += difficulty_feature_dim

        if use_streak_features:
            self.streak_dim = 8
            self.streak_projection = nn.Sequential(nn.Linear(1, self.streak_dim), nn.GELU())
            self.streak_dropout = nn.Dropout(dropout)
            input_size += self.streak_dim

        if use_time_since_last:
            self.time_since_last_dim = 8
            self.time_since_last_projection = nn.Sequential(nn.Linear(1, self.time_since_last_dim), nn.GELU())
            self.time_since_last_dropout = nn.Dropout(dropout)
            input_size += self.time_since_last_dim

        self.input_size = input_size
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.pre_transformer_norm = nn.LayerNorm(hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size, max_seq_len=400, dropout=dropout / 2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

        if use_gated_skip:
            self.skip_proj = nn.Linear(input_size, hidden_size)
            self.skip_gate_proj = nn.Linear(hidden_size * 2, hidden_size)

        self.fc = nn.Linear(hidden_size, num_concepts + 1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.noise_std = noise_std

        self.apply(self._init_weights)

        print(f"TransformerDKTModel initialized with input_size: {self.input_size}")
        print(
            f"Features enabled: time={use_time_features}, difficulty={use_difficulty}, "
            f"streak={use_streak_features}, time_since_last={use_time_since_last}"
        )
        print(
            f"Transformer config: layers={num_layers}, heads={nhead}, "
            f"dim_feedforward={dim_feedforward}, dropout={dropout}"
        )
        print(f"Regularization: noise_std={noise_std}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        concepts: torch.Tensor,
        answers: torch.Tensor,
        time_features: torch.Tensor | None = None,
        difficulty_features: torch.Tensor | None = None,
        streak_features: torch.Tensor | None = None,
        time_since_last_features: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        max_eval_length: int | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = concepts.size()

        # Optional truncation
        if max_eval_length is not None and seq_len > max_eval_length:
            seq_len = max_eval_length
            concepts = concepts[:, :max_eval_length]
            answers = answers[:, :max_eval_length]
            if mask is not None:
                mask = mask[:, :max_eval_length]
            if time_features is not None:
                time_features = time_features[:, :max_eval_length]
            if difficulty_features is not None:
                difficulty_features = difficulty_features[:, :max_eval_length]
            if streak_features is not None:
                streak_features = streak_features[:, :max_eval_length]
            if time_since_last_features is not None:
                time_since_last_features = time_since_last_features[:, :max_eval_length]

        # Concept embedding + feature dropout
        c_embed = self.concept_embedding(concepts)
        c_embed = self.feature_dropout(c_embed.unsqueeze(1)).squeeze(1)

        if self.training and self.noise_std > 0:
            noise = torch.randn_like(c_embed) * self.noise_std
            c_embed = c_embed + noise

        a_embed = answers.unsqueeze(-1)
        transformer_input = torch.cat([c_embed, a_embed], dim=-1)

        if self.use_time_features and time_features is not None:
            if self.normalize_time:
                time_embed = self.time_projection(time_features.float().unsqueeze(-1))
            else:
                time_embed = self.time_embedding(time_features)
            time_embed = self.time_dropout(time_embed)
            transformer_input = torch.cat([transformer_input, time_embed], dim=-1)

        if self.use_difficulty and difficulty_features is not None:
            diff_embed = self.difficulty_projection(difficulty_features.float().unsqueeze(-1))
            diff_embed = self.difficulty_dropout(diff_embed)
            transformer_input = torch.cat([transformer_input, diff_embed], dim=-1)

        if self.use_streak_features and streak_features is not None:
            streak_embed = self.streak_projection(streak_features.float().unsqueeze(-1))
            streak_embed = self.streak_dropout(streak_embed)
            transformer_input = torch.cat([transformer_input, streak_embed], dim=-1)

        if self.use_time_since_last and time_since_last_features is not None:
            tsl_embed = self.time_since_last_projection(time_since_last_features.float().unsqueeze(-1))
            tsl_embed = self.time_since_last_dropout(tsl_embed)
            transformer_input = torch.cat([transformer_input, tsl_embed], dim=-1)

        original_input = transformer_input

        # Dynamic input-size guard (rare)
        if transformer_input.shape[-1] != self.input_size:
            actual = transformer_input.shape[-1]
            print(f"Warning: Input size mismatch. Expected {self.input_size}, got {actual}. Adjusting projections.")
            self.input_projection = nn.Linear(actual, self.hidden_size).to(transformer_input.device)
            if self.use_gated_skip:
                self.skip_proj = nn.Linear(actual, self.hidden_size).to(transformer_input.device)
            self.input_size = actual

        x = self.input_projection(transformer_input)
        x = self.pre_transformer_norm(x)
        x = self.positional_encoding(x)

        transformer_mask = (~mask.bool()) if mask is not None else None
        x = self.transformer_encoder(x, src_key_padding_mask=transformer_mask)

        if self.use_gated_skip:
            skip_proj = self.skip_proj(original_input)
            gate_input = torch.cat([skip_proj, x], dim=-1)
            gate = torch.sigmoid(self.skip_gate_proj(gate_input))
            x = gate * x + (1 - gate) * skip_proj

        x = self.dropout(x)
        logits = self.fc(x)
        out = self.sigmoid(logits)  # returns probabilities
        return out


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class ConceptDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sequences,
        max_seq_len: int = 400,
        use_time_features: bool = False,
        normalize_time: bool = True,
        use_difficulty_features: bool = False,
        difficulty_dict=None,
        question_to_concepts=None,
        use_streak_features: bool = False,
        use_time_since_last: bool = False,
        config: TransformerConfig | None = None,
    ):
        self.sequences = sequences
        self.max_seq_len = max_seq_len
        self.use_time_features = use_time_features
        self.normalize_time = normalize_time
        self.use_difficulty_features = use_difficulty_features
        self.difficulty_dict = difficulty_dict
        self.question_to_concepts = question_to_concepts
        self.use_streak_features = use_streak_features
        self.use_time_since_last = use_time_since_last
        self.config = config
        self.training = False  # Can be toggled externally

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        c_seq, a_seq = self.sequences[idx]

        # Optional cropping augmentation
        is_training = getattr(self, "training", False)
        if is_training and getattr(self.config, "use_sequence_cropping", False) and len(c_seq) > 10:
            crop_size = random.randint(0, min(5, int(len(c_seq) * self.config.crop_ratio)))
            if crop_size > 0:
                c_seq = c_seq[crop_size:]
                a_seq = a_seq[crop_size:]

        if isinstance(a_seq, list):
            a_seq = np.array([float(a) for a in a_seq])

        # Time features
        time_features = None
        if self.use_time_features:
            seq_len = len(c_seq)
            raw_time_gaps = np.zeros(seq_len)
            for i in range(1, seq_len):
                prev_answer = float(a_seq[i - 1])
                base_time = np.random.uniform(20, 60) if prev_answer > 0.5 else np.random.uniform(120, 300)
                noise_factor = np.random.uniform(0.95, 1.05)
                raw_time_gaps[i] = base_time * noise_factor

            if self.normalize_time:
                max_time = np.max(raw_time_gaps) + 1e-6
                time_features = raw_time_gaps / max_time
                if is_training and getattr(self.config, "add_feature_noise", False):
                    noise = np.random.uniform(-self.config.noise_level, self.config.noise_level, size=time_features.shape)
                    time_features = np.clip(time_features + noise, 0.0, 1.0)
            else:
                time_buckets = np.zeros(seq_len, dtype=np.int64)
                for i in range(seq_len):
                    if raw_time_gaps[i] <= 30:
                        time_buckets[i] = 0
                    elif raw_time_gaps[i] <= 300:
                        time_buckets[i] = 1
                    else:
                        time_buckets[i] = 2
                time_features = time_buckets

        # Streak features
        streak_features = None
        if self.use_streak_features:
            seq_len = len(c_seq)
            streak_features = np.zeros(seq_len)
            current_streak = 0
            for i in range(seq_len):
                if i > 0:
                    prev_correct = float(a_seq[i - 1]) > 0.5
                    curr_correct = float(a_seq[i]) > 0.5
                    if prev_correct == curr_correct:
                        current_streak += 1
                    else:
                        current_streak = 0
                streak_features[i] = current_streak
            streak_features = np.minimum(streak_features, 10) / 10.0

        # Time-since-last features
        time_since_last_features = None
        if self.use_time_since_last:
            seq_len = len(c_seq)
            time_since_last_features = np.zeros(seq_len)
            last_seen = {}
            for i, concept in enumerate(c_seq):
                if concept in last_seen:
                    time_since_last_features[i] = i - last_seen[concept]
                else:
                    time_since_last_features[i] = 0
                last_seen[concept] = i
            max_time = np.max(time_since_last_features) + 1e-6
            time_since_last_features = time_since_last_features / max_time

        # Difficulty features
        difficulty_features = None
        if self.use_difficulty_features and self.difficulty_dict is not None:
            seq_len = len(c_seq)
            difficulty_features = np.zeros(seq_len)

            if self.question_to_concepts is not None:
                concept_to_questions = defaultdict(list)
                for q_id, concept in self.question_to_concepts.items():
                    concept_to_questions[concept].append(q_id)

                for i, concept in enumerate(c_seq):
                    str_concept = str(concept)
                    question_ids = concept_to_questions.get(concept, []) or concept_to_questions.get(str_concept, [])
                    if question_ids:
                        q_id = question_ids[0]
                        difficulty_features[i] = self.difficulty_dict.get(q_id, 0.5)
                    else:
                        difficulty_features[i] = 0.5
            else:
                for i, concept in enumerate(c_seq):
                    if isinstance(concept, (int, float)):
                        difficulty = self.difficulty_dict.get(concept, None)
                        if difficulty is None:
                            difficulty = self.difficulty_dict.get(str(concept), 0.5)
                    else:
                        difficulty = self.difficulty_dict.get(concept, 0.5)
                    difficulty_features[i] = difficulty

        # Truncate/pad
        if len(c_seq) > self.max_seq_len:
            c_seq = c_seq[: self.max_seq_len]
            a_seq = a_seq[: self.max_seq_len]
            if time_features is not None:
                time_features = time_features[: self.max_seq_len]
            if difficulty_features is not None:
                difficulty_features = difficulty_features[: self.max_seq_len]
            if streak_features is not None:
                streak_features = streak_features[: self.max_seq_len]
            if time_since_last_features is not None:
                time_since_last_features = time_since_last_features[: self.max_seq_len]

        seq_len = len(c_seq)
        result = {
            "concepts": torch.LongTensor(c_seq),
            "answers": torch.FloatTensor(a_seq),
            "seq_len": seq_len,
        }

        if time_features is not None:
            if self.normalize_time:
                result["time_features"] = torch.FloatTensor(time_features)
            else:
                result["time_features"] = torch.LongTensor(time_features)

        if difficulty_features is not None:
            result["difficulty_features"] = torch.FloatTensor(difficulty_features)

        if streak_features is not None:
            result["streak_features"] = torch.FloatTensor(streak_features)

        if time_since_last_features is not None:
            result["time_since_last_features"] = torch.FloatTensor(time_since_last_features)

        return result


def collate_fn(batch):
    """Pad variable-length sequences and stack features."""
    batch = sorted(batch, key=lambda x: x["seq_len"], reverse=True)
    seq_lens = [x["seq_len"] for x in batch]
    max_len = max(seq_lens)

    concepts = torch.zeros(len(batch), max_len, dtype=torch.long)
    answers = torch.zeros(len(batch), max_len, dtype=torch.float)
    mask = torch.zeros(len(batch), max_len, dtype=torch.float)

    has_time = "time_features" in batch[0]
    if has_time:
        time_dtype = batch[0]["time_features"].dtype
        time_features = torch.zeros(len(batch), max_len, dtype=time_dtype)

    has_difficulty = "difficulty_features" in batch[0]
    if has_difficulty:
        difficulty_features = torch.zeros(len(batch), max_len, dtype=torch.float)

    has_streak = "streak_features" in batch[0]
    if has_streak:
        streak_features = torch.zeros(len(batch), max_len, dtype=torch.float)

    has_tsl = "time_since_last_features" in batch[0]
    if has_tsl:
        time_since_last_features = torch.zeros(len(batch), max_len, dtype=torch.float)

    for i, item in enumerate(batch):
        sl = item["seq_len"]
        concepts[i, :sl] = item["concepts"]
        answers[i, :sl] = item["answers"]
        mask[i, :sl] = 1
        if has_time:
            time_features[i, :sl] = item["time_features"]
        if has_difficulty:
            difficulty_features[i, :sl] = item["difficulty_features"]
        if has_streak:
            streak_features[i, :sl] = item["streak_features"]
        if has_tsl:
            time_since_last_features[i, :sl] = item["time_since_last_features"]

    result = {
        "concepts": concepts,
        "answers": answers,
        "mask": mask,
        "seq_lens": torch.LongTensor(seq_lens),
    }
    if has_time:
        result["time_features"] = time_features
    if has_difficulty:
        result["difficulty_features"] = difficulty_features
    if has_streak:
        result["streak_features"] = streak_features
    if has_tsl:
        result["time_since_last_features"] = time_since_last_features
    return result


# -----------------------------------------------------------------------------
# Training and Evaluation
# -----------------------------------------------------------------------------
def train_transformer_model(
    model,
    train_loader,
    val_loader,
    config: TransformerConfig,
    fold: int | None = None,
    teacher_model=None,
    optimizer_params: dict | None = None,
    fast_eval: bool = False,
    sample_size: int = 100,
):
    print("\nStarting transformer model training...")
    log_dir_root = getattr(config, "tensorboard_dir", "tensorboard_logs")
    tensorboard_log_dir = os.path.join(log_dir_root, f"fold_{fold}" if fold is not None else "default")
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    print(f"TensorBoard logs will be available at: {tensorboard_log_dir}")

    criterion = SmoothBCEwLogits(smoothing=config.label_smoothing)

    if optimizer_params is None:
        optimizer_params = {"lr": config.learning_rate, "weight_decay": config.weight_decay}
    optimizer = optim.AdamW(model.parameters(), **optimizer_params)

    if config.use_cosine_scheduler:
        t_mult = getattr(config, "cosine_t_mult", 2)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config.cosine_restart_epochs, T_mult=t_mult, eta_min=1e-6
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, verbose=True)

    scaler = GradScaler(enabled=config.use_mixed_precision and torch.cuda.is_available())

    swa_model = None
    swa_scheduler = None
    if config.use_swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=config.swa_lr, anneal_epochs=config.swa_anneal_epochs)
        print(f"Using SWA starting from epoch {config.swa_start}")

    train_losses, val_aucs, val_accs, epochs = [], [], [], []
    best_val_auc = 0.0
    patience_counter = 0

    fold_suffix = f"_fold_{fold}" if fold is not None else ""
    model_save_dir = os.path.join(getattr(config, "weights_dir", "weights"), f"model{fold_suffix}")
    os.makedirs(model_save_dir, exist_ok=True)

    if teacher_model is not None and config.use_knowledge_distillation:
        teacher_model.eval()
        print("Using knowledge distillation with teacher model")

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, total=len(train_loader), disable=False, ncols=80, leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            concepts = batch["concepts"].to(device)
            answers = batch["answers"].to(device)
            mask = batch["mask"].to(device)
            seq_lens = batch.get("seq_lens", None)
            if seq_lens is not None:
                seq_lens = seq_lens.to(device)

            # CutMix (simple version on first sample of batch)
            if config.use_cutmix and random.random() < config.cutmix_prob:
                rand_idx = random.randint(0, len(train_loader.dataset) - 1)
                rand_batch = train_loader.dataset[rand_idx]
                rand_concepts = rand_batch["concepts"].to(device)
                rand_answers = rand_batch["answers"].to(device)

                seq_len = min(concepts.size(1), rand_concepts.size(0))
                if seq_len > 10:
                    cut_start = random.randint(0, seq_len - 5)
                    cut_length = random.randint(3, min(10, seq_len - cut_start))
                    if concepts.size(0) > 0 and rand_concepts.size(0) >= cut_length:
                        concepts[0, cut_start : cut_start + cut_length] = rand_concepts[:cut_length]
                        answers[0, cut_start : cut_start + cut_length] = rand_answers[:cut_length]

            time_features = batch.get("time_features", None)
            if time_features is not None:
                time_features = time_features.to(device)

            difficulty_features = batch.get("difficulty_features", None)
            if difficulty_features is not None:
                difficulty_features = difficulty_features.to(device)

            streak_features = batch.get("streak_features", None)
            if streak_features is not None:
                streak_features = streak_features.to(device)

            tsl_features = batch.get("time_since_last_features", None)
            if tsl_features is not None:
                tsl_features = tsl_features.to(device)

            grad_accum_steps = getattr(config, "grad_accum_steps", 1)
            if batch_idx % grad_accum_steps == 0:
                optimizer.zero_grad()

            with autocast(enabled=config.use_mixed_precision and torch.cuda.is_available()):
                pred = model(
                    concepts,
                    answers,
                    time_features,
                    difficulty_features,
                    streak_features,
                    tsl_features,
                    mask,
                    seq_lens,
                )

                target_concepts = concepts[:, 1:]
                target_answers = answers[:, 1:]
                target_mask = mask[:, 1:] if mask is not None else None

                pred = pred[:, :-1]
                bsz, sl = target_concepts.size()
                b_idx = torch.arange(bsz).unsqueeze(1).expand(-1, sl).to(device)
                s_idx = torch.arange(sl).unsqueeze(0).expand(bsz, -1).to(device)

                pred_gathered = pred[b_idx, s_idx, target_concepts]

                valid_indices = target_mask > 0 if target_mask is not None else torch.ones_like(
                    target_answers, dtype=torch.bool
                )

                if valid_indices.sum() > 0:
                    student_loss = criterion(pred_gathered[valid_indices], target_answers[valid_indices])
                    loss = student_loss

                    if teacher_model is not None and config.use_knowledge_distillation:
                        with torch.no_grad():
                            t_pred = teacher_model(
                                concepts, answers, time_features, difficulty_features, streak_features, tsl_features, mask, seq_lens
                            )
                            t_pred = t_pred[:, :-1]
                            t_pred_gathered = t_pred[b_idx, s_idx, target_concepts]

                        temp = config.distillation_temp
                        s_logits = torch.log(pred_gathered[valid_indices] / (1 - pred_gathered[valid_indices] + 1e-7)) / temp
                        t_logits = torch.log(t_pred_gathered[valid_indices] / (1 - t_pred_gathered[valid_indices] + 1e-7)) / temp
                        s_probs = torch.sigmoid(s_logits)
                        t_probs = torch.sigmoid(t_logits)

                        distill_loss = F.kl_div(torch.log(s_probs + 1e-7), t_probs, reduction="batchmean") * (temp * temp)
                        alpha = config.distillation_alpha
                        loss = (1 - alpha) * student_loss + alpha * distill_loss

                    l2_reg = 0.0
                    for p in model.parameters():
                        l2_reg += torch.norm(p, p=2)
                    loss += config.l2_reg * l2_reg

                    loss = loss / grad_accum_steps

            if valid_indices.sum() > 0:
                scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
                scaler.step(optimizer)
                scaler.update()

            epoch_loss += (loss.item() * grad_accum_steps) if valid_indices.sum() > 0 else 0.0
            progress_bar.set_description("Training")

        epoch_loss /= max(1, len(train_loader))
        train_losses.append(epoch_loss)
        epochs.append(epoch + 1)
        print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if config.use_swa and epoch >= config.swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            # For CosineAnnealingWarmRestarts we step every epoch
            if isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
                scheduler.step(epoch + 1)
            else:
                scheduler.step(val_aucs[-1] if len(val_aucs) else 0)

        print("Evaluating on validation set...")
        val_auc, val_acc = evaluate_transformer_model(model, val_loader, fast_eval=fast_eval, sample_size=sample_size)
        val_aucs.append(val_auc)
        val_accs.append(val_acc)
        print(f"Epoch {epoch + 1} validation AUC: {val_auc:.4f}, Accuracy: {val_acc:.4f}")

        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("AUC/validation", val_auc, epoch)
        writer.add_scalar("Accuracy/validation", val_acc, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_path = os.path.join(model_save_dir, f"best_transformer_model_fold_{fold}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": val_auc,
                    "val_acc": val_acc,
                    "epoch": epoch,
                    "fold": fold,
                    "config": {k: v for k, v in vars(config).items() if not callable(v)},
                },
                best_model_path,
            )
            print(f"Saved best model with validation AUC: {val_auc:.4f}")
        else:
            patience_counter += 1
            if config.early_stopping and patience_counter >= config.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    if config.use_swa and (epoch >= config.swa_start):
        print("\nUpdating batch normalization statistics for SWA model...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model)
        print("Evaluating SWA model on validation set...")
        swa_val_auc, swa_val_acc = evaluate_transformer_model(swa_model, val_loader, fast_eval=fast_eval, sample_size=sample_size)
        print(f"SWA model validation AUC: {swa_val_auc:.4f}, Accuracy: {swa_val_acc:.4f}")
        if swa_val_auc > best_val_auc:
            print("SWA model performs better, saving as best model")
            torch.save(
                {
                    "model_state_dict": swa_model.state_dict(),
                    "val_auc": swa_val_auc,
                    "val_acc": swa_val_acc,
                    "epoch": epoch,
                    "fold": fold,
                    "config": {k: v for k, v in vars(config).items() if not callable(v)},
                },
                os.path.join(model_save_dir, "best_transformer_model_swa.pt"),
            )
            best_val_auc = swa_val_auc

        writer.add_scalar("AUC/validation_swa", swa_val_auc, epoch)
        writer.add_scalar("Accuracy/validation_swa", swa_val_acc, epoch)

    writer.close()
    print("Training completed!")
    return train_losses, val_aucs, best_val_auc


def evaluate_transformer_model(model, data_loader, fast_eval=False, sample_size=100, max_eval_length=200):
    """
    Evaluate the Transformer DKT model using optional sub-sampling and truncation.
    Mirrors the training-time gather: compare p(correct | next concept) vs actual next answer.
    Returns: (auc, acc)
    """
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score, accuracy_score

    model.eval()
    device = next(model.parameters()).device
    y_true, y_prob = [], []

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if fast_eval and i >= sample_size:
                break

            # collate_fn returns a dict
            concepts = batch["concepts"].to(device)              # [B, T]
            answers  = batch["answers"].to(device).float()       # [B, T]
            mask     = batch.get("mask", None)
            if mask is not None:
                mask = mask.to(device)                           # [B, T]
            seq_lens = batch.get("seq_lens", None)
            if seq_lens is not None:
                seq_lens = seq_lens.to(device)

            kw = {}
            for name in ["time_features", "difficulty_features", "streak_features", "time_since_last_features"]:
                if name in batch and batch[name] is not None:
                    kw[name] = batch[name].to(device)

            # Truncate sequence length on-the-fly for speed
            if isinstance(max_eval_length, int) and max_eval_length > 0 and concepts.size(1) > max_eval_length:
                concepts = concepts[:, :max_eval_length]
                answers  = answers[:,  :max_eval_length]
                if mask is not None:
                    mask = mask[:, :max_eval_length]
                for name in ["time_features","difficulty_features","streak_features","time_since_last_features"]:
                    if name in kw and kw[name] is not None and kw[name].size(1) > max_eval_length:
                        kw[name] = kw[name][:, :max_eval_length]

            # Forward pass — model already returns probabilities in shape [B, T, num_concepts+1]
            probs_all = model(concepts, answers, mask=mask, seq_lens=seq_lens, max_eval_length=max_eval_length, **kw)

            # Compare p(correct @ t | concept at t) to actual answer at t (shift by 1 like training)
            target_concepts = concepts[:, 1:]        # [B, T-1]
            target_answers  = answers[:,  1:]        # [B, T-1]
            target_mask     = mask[:,    1:] if mask is not None else torch.ones_like(target_answers)

            # Align predictions to t (drop last time step)
            probs_all = probs_all[:, :-1, :]         # [B, T-1, C+1]

            # Gather probability for the actual next concept
            bsz, sl = target_concepts.size()
            b_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(-1, sl)
            s_idx = torch.arange(sl,  device=device).unsqueeze(0).expand(bsz, -1)
            probs = probs_all[b_idx, s_idx, target_concepts]  # [B, T-1]

            valid = target_mask > 0
            if valid.any():
                y_true.extend(target_answers[valid].detach().cpu().numpy().tolist())
                y_prob.extend(probs[valid].detach().cpu().numpy().tolist())

    if len(set(y_true)) < 2:
        # Not enough class variety to compute AUC
        try:
            pred = (np.array(y_prob) >= 0.5).astype(int)
            acc = (pred == np.array(y_true, dtype=int)).mean() if len(y_true) else 0.0
        except Exception:
            acc = 0.0
        return 0.0, float(acc)

    try:
        auc = float(roc_auc_score(y_true, y_prob))
        pred = (np.array(y_prob) >= 0.5).astype(int)
        acc = float(accuracy_score(y_true, pred))
        return auc, acc
    except Exception:
        return 0.0, 0.0


def create_transformer_dataset(sequences, concept_map, config: TransformerConfig):
    """
    Build a ConceptDataset with optional difficulty dictionary.
    `concept_map` can be used to size or map concepts (kept for compatibility).
    """
    import numpy as np

    difficulty_dict = None
    if getattr(config, "use_difficulty", False):
        try:
            # Try to build from external metadata if available
            difficulty_dict = build_difficulty_dictionary(getattr(config, "metadata_path", None))
        except Exception as e:
            print(f"Error building difficulty dictionary: {e}")
            print("Using synthetic difficulty scores instead.")
            difficulty_dict = {}
            # Seed both int and str keys for robustness
            for concept_id in range(1, len(concept_map) + 1 if concept_map else 1000):
                v = float(np.random.random())
                difficulty_dict[concept_id] = v
                difficulty_dict[str(concept_id)] = v

    dataset = ConceptDataset(
        sequences,
        max_seq_len=config.max_seq_len,
        use_time_features=config.use_time_features,
        normalize_time=config.normalize_time,
        use_difficulty_features=config.use_difficulty,
        difficulty_dict=difficulty_dict,
        use_streak_features=config.use_streak_features,
        use_time_since_last=config.use_time_since_last,
        config=config,
    )
    return dataset


# -----------------------------------------------------------------------------
# Main (demo)
# -----------------------------------------------------------------------------
def main():
    print("Transformer-based Deep Knowledge Tracing (DKT) Model")
    print("==================================================")

    config = TransformerConfig()
    random_seed = random.randint(1, 10000) if config.random_seed is None else config.random_seed

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    print(f"Random seed set to: {random_seed}")

    print("\nConfiguration:")
    for attr, value in vars(config).items():
        print(f"  {attr}: {value}")

    print("\nThis is a standalone transformer-based DKT model implementation.")
    print("To use it, import the TransformerDKTModel class and related functions.")
    print("Example usage:")
    print("  from transformer_dkt_model import TransformerDKTModel, TransformerConfig")
    print("  model = TransformerDKTModel(num_concepts=100)")


# Backward-compatibility alias for existing scripts:
TransformerDKT = TransformerDKTModel

if __name__ == "__main__":
    main()
