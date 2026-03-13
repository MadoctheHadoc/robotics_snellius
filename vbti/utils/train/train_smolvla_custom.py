"""
Train SmolVLA on lift_cube_3cams dataset.

Usage:
    python examples/training/train_smolvla_lift_cube.py
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
import torch
from tqdm import tqdm

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

from vbti.utils.datasets import load_and_split_dataset, create_dataloaders


def validate(policy, val_loader, preprocessor, device, val_size):
    """
    Run validation on a subsample of validation set.

    Args:
        policy: The policy model
        val_loader: Validation dataloader
        preprocessor: Batch preprocessor
        device: torch device
        val_size: Number of batches to sample

    Returns:
        Average validation loss
    """
    policy.eval()
    val_losses = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= val_size:
                break

            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            val_losses.append(loss.item())

    policy.train()

    if val_losses:
        return sum(val_losses) / len(val_losses)
    return float('inf')


def main():
    # ============== CONFIG ==============
    dataset_repo_id = "eternalmay33/lift_cube_3cams"
    # Local path to the dataset root directory (parent of eternalmay33/lift_cube_3cams/).
    # Set to None to use the default HuggingFace cache (~/.cache/huggingface/lerobot).
    data_dir = Path("/gpfs/home2/lpopdimitrova/madoc/robotics_snellius/data")
    output_directory = Path("outputs/train/smolvla_lift_cube_3cams_v2")
    output_directory.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training config
    training_steps = 10000
    batch_size = 4  # SmolVLA is memory-heavy, adjust based on your GPU
    log_freq = 100
    save_freq = 1000
    val_freq = 500  # Run validation every N steps
    val_size = 50   # Number of batches to sample from val set each validation
    learning_rate = 1e-5

    # SmolVLA config
    chunk_size = 50  # Predict 50 future actions
    n_obs_steps = 1  # Use only current observation

    print("=" * 60)
    print("SMOLVLA TRAINING")
    print("=" * 60)
    print(f"Dataset: {dataset_repo_id}")
    print(f"Output: {output_directory}")
    print(f"Device: {device}")
    print(f"Training steps: {training_steps}")
    print(f"Batch size: {batch_size}")
    print("=" * 60)

    # ============== LOAD DATASET METADATA ==============
    print("\nLoading dataset metadata...")
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id, root=data_dir)

    print(f"Total episodes: {dataset_metadata.total_episodes}")
    print(f"Total frames: {dataset_metadata.total_frames}")
    print(f"FPS: {dataset_metadata.fps}")
    print(f"Features: {list(dataset_metadata.features.keys())}")

    # ============== CONFIGURE POLICY ==============
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    print(f"\nInput features: {list(input_features.keys())}")
    print(f"Output features: {list(output_features.keys())}")

    # SmolVLA configuration
    cfg = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=n_obs_steps,
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        # Finetuning settings - freeze most of the model
        freeze_vision_encoder=True,
        train_expert_only=True,
        train_state_proj=True,
        # Optimizer settings
        optimizer_lr=learning_rate,
        optimizer_weight_decay=1e-10,
        optimizer_grad_clip_norm=10.0,
        # Scheduler
        scheduler_warmup_steps=500,
        scheduler_decay_steps=training_steps,
    )

    # ============== CREATE DELTA TIMESTAMPS ==============
    # SmolVLA needs specific timestamps for observations and actions
    fps = dataset_metadata.fps

    delta_timestamps = {
        # Observations: only current frame (index 0)
        "observation.state": [i / fps for i in cfg.observation_delta_indices],
    }

    # Add timestamps for all image features
    for key in input_features:
        if input_features[key].type is FeatureType.VISUAL:
            delta_timestamps[key] = [i / fps for i in cfg.observation_delta_indices]

    # Actions: chunk_size future actions
    delta_timestamps["action"] = [i / fps for i in cfg.action_delta_indices]

    print(f"\nDelta timestamps:")
    for key, vals in delta_timestamps.items():
        print(f"  {key}: {len(vals)} frames")

    # ============== LOAD DATASET ==============
    print("\nLoading dataset with train/val split...")

    full_dataset, train_dataset, val_dataset = load_and_split_dataset(
        repo_id=dataset_repo_id,
        root=data_dir,
        delta_timestamps=delta_timestamps,
        train_ratio=1.0,
        verbose=True,
    )

    print(f"Train dataset: {len(train_dataset)} frames")
    print(f"Val dataset: {len(val_dataset) if val_dataset else 0} frames")

    # ============== INITIALIZE POLICY ==============
    print("\nInitializing SmolVLA from pretrained base...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base", config=cfg)
    policy.train()
    policy.to(device)

    # Count trainable parameters
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    # Create preprocessor/postprocessor
    preprocessor, postprocessor = make_smolvla_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    # ============== OPTIMIZER & DATALOADER ==============
    optimizer_config = cfg.get_optimizer_preset()
    optimizer = optimizer_config.build(policy.parameters())

    if val_dataset is not None:
        train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, batch_size=batch_size, num_workers=0)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = None

    # ============== TRAINING LOOP ==============
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    step = 0
    done = False
    losses = []
    best_val_loss = float('inf')

    while not done:
        for batch in tqdm(train_loader, desc=f"Step {step}", leave=False):
            # Preprocess
            batch = preprocessor(batch)

            # Forward pass
            loss, loss_dict = policy.forward(batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                cfg.optimizer_grad_clip_norm
            )

            optimizer.step()

            losses.append(loss.item())
            step += 1

            # Logging
            if step % log_freq == 0:
                avg_loss = sum(losses[-log_freq:]) / min(len(losses), log_freq)
                print(f"Step {step:5d} | Train Loss: {avg_loss:.4f}")

            # Validation (only if we have a validation set)
            if val_loader is not None and step % val_freq == 0:
                val_loss = validate(policy, val_loader, preprocessor, device, val_size)
                print(f"Step {step:5d} | Val Loss: {val_loss:.4f}")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_dir = output_directory / "best"
                    print(f"  New best! Saving to {best_dir}")
                    policy.save_pretrained(best_dir)
                    preprocessor.save_pretrained(best_dir)
                    postprocessor.save_pretrained(best_dir)

            # Save checkpoint
            if step % save_freq == 0:
                checkpoint_dir = output_directory / f"checkpoint_{step:06d}"
                print(f"\nSaving checkpoint to {checkpoint_dir}")
                policy.save_pretrained(checkpoint_dir)
                preprocessor.save_pretrained(checkpoint_dir)
                postprocessor.save_pretrained(checkpoint_dir)

            if step >= training_steps:
                done = True
                break

    # ============== SAVE FINAL MODEL ==============
    final_dir = output_directory / "final"
    print(f"\nSaving final model to {final_dir}")
    policy.save_pretrained(final_dir)
    preprocessor.save_pretrained(final_dir)
    postprocessor.save_pretrained(final_dir)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final train loss: {losses[-1]:.4f}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Final model: {final_dir}")
    print(f"Best model: {output_directory / 'best'}")


if __name__ == "__main__":
    main()
