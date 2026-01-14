import torch
from pathlib import Path
import hydra
from diffusion_policy.workspace.train_diffusion_unet_lowdim_workspace import TrainDiffusionUnetLowdimWorkspace

ckpt = Path("diffusion_policy/data/outputs/2026.01.14/11.41.26_train_diffusion_unet_lowdim_guide_guide_lowdim/checkpoints/epoch=0240-test_mean_score=0.877.ckpt")
ws = TrainDiffusionUnetLowdimWorkspace.create_from_checkpoint(str(ckpt))
ws.cfg.task.dataset.val_ratio = 0.2
ws.cfg.task.env_runner.val_ratio = 0.2
ws.cfg.task.env_runner.n_test = 20

policy = ws.ema_model if getattr(ws, "ema_model", None) is not None else ws.model
policy.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

runner = hydra.utils.instantiate(ws.cfg.task.env_runner, output_dir="diffusion_policy/data/outputs/eval_guide_lowdim")
print(runner.run(policy))
