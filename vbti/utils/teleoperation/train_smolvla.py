"""
Minimal SmolVLA finetuning on the lidia552/Embodia_project HuggingFace dataset.

The dataset is gated — put your token in .env:
    HF_TOKEN=hf_...

Run from repo root:
    python vbti/utils/teleoperation/train_smolvla.py
    python vbti/utils/teleoperation/train_smolvla.py --output_dir outputs/smolvla --total_steps 2000
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

REPO_ID = "lidia552/Embodia_project"
CHUNK_SIZE = 50
LEARNING_RATE = 1e-5
BATCH_SIZE = 4
TOTAL_STEPS = 500
LOG_FREQ = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune SmolVLA on lidia552/Embodia_project")
    parser.add_argument("--repo_id", type=str, default=REPO_ID)
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

    # Load HuggingFace token from .env
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("Logged in to HuggingFace via .env token")
    else:
        print("No HF_TOKEN in .env — assuming already logged in via huggingface-cli")

    print("=" * 60)
    print("SmolVLA Finetuning — SO-101 / Embodia_project")
    print("=" * 60)
    print(f"Dataset  : {args.repo_id}")
    print(f"Output   : {args.output_dir}")
    print(f"Device   : {device}")
    print(f"Steps    : {args.total_steps}  batch={args.batch_size}  lr={args.lr}")
    print()

    # Download dataset from HuggingFace into a local cache directory.
    # Each episode is its own sub-folder (ep2/, ep3/, …) so we use snapshot_download
    # and treat the local dir as the episodes root, bypassing lerobot's version-tag check.
    episodes_dir = Path.home() / ".cache/huggingface/lerobot_embodia"
    if not any(episodes_dir.iterdir()) if episodes_dir.exists() else True:
        print(f"Downloading dataset to {episodes_dir} ...")
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=str(episodes_dir),
            token=hf_token,
        )
        print("Download complete.\n")
    else:
        print(f"Using cached dataset at {episodes_dir}\n")

    # Discover episode sub-folders (ep2, ep3, …)
    episode_dirs = sorted([d for d in episodes_dir.iterdir() if d.is_dir() and (d / "meta" / "info.json").exists()])
    assert episode_dirs, f"No episode folders found in {episodes_dir}"
    print(f"Found {len(episode_dirs)} episodes: {[d.name for d in episode_dirs]}")

    # Load metadata from every episode and take the intersection of image features.
    # Episodes can have different camera names (e.g. 'downh' vs 'down'), so using
    # only the first episode's keys would break loading for the others.
    all_metas = [
        LeRobotDatasetMetadata(repo_id=ep_dir.name, root=ep_dir)
        for ep_dir in episode_dirs
    ]
    first_meta = all_metas[0]
    fps = first_meta.fps

    # Start from the first episode's full feature set, then keep only keys present in all
    features = dataset_to_policy_features(first_meta.features)
    output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
    input_features = {k: v for k, v in features.items() if k not in output_features}

    all_image_keys = [set(
        k for k in dataset_to_policy_features(m.features) if "images" in k
    ) for m in all_metas]
    common_image_keys = set.intersection(*all_image_keys)
    # Drop image keys not shared by every episode
    input_features = {
        k: v for k, v in input_features.items()
        if "images" not in k or k in common_image_keys
    }

    print(f"Input features  : {list(input_features)}")
    print(f"Output features : {list(output_features)}")
    print(f"FPS             : {fps}")
    if common_image_keys != all_image_keys[0]:
        dropped = all_image_keys[0] - common_image_keys
        print(f"Dropped non-common camera keys: {dropped}")
    print()

    # Delta timestamps: current frame for observations, next chunk_size frames for actions
    delta_timestamps = {
        "observation.state": [0.0],
        "action": [i / fps for i in range(args.chunk_size)],
    }
    for key in common_image_keys:
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

    # Load all episodes and combine. root=ep_dir so that lerobot finds meta/info.json
    # at ep_dir/meta/info.json without falling back to a HuggingFace lookup.
    datasets = []
    for ep_dir in episode_dirs:
        ds = LeRobotDataset(
            repo_id=ep_dir.name,
            root=ep_dir,
            delta_timestamps=delta_timestamps,
            video_backend="pyav",
        )
        datasets.append(ds)
        print(f"  {ep_dir.name}: {len(ds)} frames")

    combined = ConcatDataset(datasets)
    loader = DataLoader(combined, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"\nTotal training frames: {len(combined)}")

    # Load SmolVLA base model
    print("\nLoading SmolVLA base model (lerobot/smolvla_base)...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base", config=cfg)
    policy.train()
    policy.to(device)

    # Preprocessor uses stats from first episode for normalization
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

    # Save finetuned model
    policy.save_pretrained(str(args.output_dir))
    print(f"\nSaved finetuned model to {args.output_dir}")
    print("Load it later with: SmolVLAPolicy.from_pretrained('<output_dir>')")


if __name__ == "__main__":
    main()
