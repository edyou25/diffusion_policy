import os
import sys
import warnings

# 过滤 pkg_resources 弃用警告（来自 llvmlite）
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*", category=UserWarning)

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from diffusion_policy.dataset.guide_lowdim_dataset import GuideLowdimDataset


def test():
    data_dir = os.environ.get("GUIDE_DATA_DIR", "/home/yyf/IROS2026/FollowDataset/data")
    if not os.path.exists(data_dir):
        print(f"[warn] data_dir not found: {data_dir}")
        return
    dataset = GuideLowdimDataset(
        data_dir=data_dir,
        horizon=16,
        pad_before=1,
        pad_after=7,
        val_ratio=0.05,
        max_train_episodes=10,
    )
    print("train size:", len(dataset))
    sample = dataset[0]
    print("obs shape:", sample["obs"].shape)
    print("action shape:", sample["action"].shape)

    val_dataset = dataset.get_validation_dataset()
    print("val size:", len(val_dataset))

    normalizer = dataset.get_normalizer()
    print("normalizer keys:", list(normalizer.params_dict.keys()))


if __name__ == "__main__":
    test()
