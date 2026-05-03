"""
Minimal SmolVLA finetuning on locally saved SO-101 episodes.

Run from repo root:
    python vbti/utils/teleoperation/train_smolvla.py
    python vbti/utils/teleoperation/train_smolvla.py --episodes_dir saved_episodes --output_dir outputs/smolvla
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

CHUNK_SIZE = 50
LEARNING_RATE = 1e-5
BATCH_SIZE = 4
TOTAL_STEPS = 500
LOG_FREQ = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune SmolVLA on saved SO-101 episodes")
    parser.add_argument("--episodes_dir", type=Path, default=Path("saved_episodes"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/smolvla_so101"))
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--total_steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("SmolVLA Finetuning — SO-101 Episodes")
    print("=" * 60)
    print(f"Episodes dir : {args.episodes_dir}")
    print(f"Output dir   : {args.output_dir}")
    print(f"Device       : {device}")
    print(f"Steps        : {args.total_steps}  batch={args.batch_size}  lr={args.lr}")
    print()

    # Discover episode folders (ep2, ep3, ...)
    episode_dirs = sorted([d for d in args.episodes_dir.iterdir() if d.is_dir()])
    assert episode_dirs, f"No episode folders found in {args.episodes_dir}"
    print(f"Found {len(episode_dirs)} episodes: {[d.name for d in episode_dirs]}")

    # Load metadata from first episode to derive features and fps
    first_meta = LeRobotDatasetMetadata(
        repo_id=episode_dirs[0].name, root=args.episodes_dir
    )
    features = dataset_to_policy_features(first_meta.features)
    output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    input_features = {k: v for k, v in features.items() if k not in output_features}
    fps = first_meta.fps

    print(f"State/action dim : {list(input_features)}")
    print(f"Action features  : {list(output_features)}")
    print(f"FPS              : {fps}")
    print()

    # Delta timestamps: single current frame for observations, chunk of future actions
    image_keys = [k for k in input_features if "images" in k]
    delta_timestamps = {"observation.state": [0.0], "action": [i / fps for i in range(args.chunk_size)]}
    for key in image_keys:
        delta_timestamps[key] = [0.0]

    # SmolVLA config
    cfg = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=1,
        chunk_size=args.chunk_size,
        freeze_vision_encoder=True,
        train_expert_only=True,
        train_state_proj=True,
        optimizer_lr=args.lr,
        optimizer_weight_decay=1e-10,
        optimizer_grad_clip_norm=10.0,
        device=str(device),
    )

    # Load all episodes and combine into one dataset
    datasets = []
    for ep_dir in episode_dirs:
        ds = LeRobotDataset(
            repo_id=ep_dir.name,
            root=args.episodes_dir,
            delta_timestamps=delta_timestamps,
        )
        datasets.append(ds)
        print(f"  {ep_dir.name}: {len(ds)} frames")

    combined = ConcatDataset(datasets)
    loader = DataLoader(combined, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"\nTotal training frames: {len(combined)}")

    # Load SmolVLA base model from HuggingFace
    print("\nLoading SmolVLA base model (lerobot/smolvla_base)...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base", config=cfg)
    policy.train()
    policy.to(device)

    # Preprocessor handles normalization using stats from the first episode.
    # For a multi-episode setup with different stats, consider aggregating stats first.
    preprocessor, _ = make_smolvla_pre_post_processors(cfg, dataset_stats=first_meta.stats)

    optimizer = cfg.get_optimizer_preset().build(policy.parameters())

    # Training loop
    print(f"\nTraining for {args.total_steps} steps...")
    print("-" * 60)
    global_step = 0

    while global_step < args.total_steps:
        for batch in loader:
            if global_step >= args.total_steps:
                break

            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            optimizer.step()

            global_step += 1
            if global_step % LOG_FREQ == 0:
                print(f"step {global_step:>5}/{args.total_steps}  loss={loss.item():.4f}")

    # Save finetuned model in HuggingFace pretrained format
    policy.save_pretrained(str(args.output_dir))
    print(f"\nSaved finetuned model to {args.output_dir}")
    print("Load it later with: SmolVLAPolicy.from_pretrained('<output_dir>')")


if __name__ == "__main__":
    main()
