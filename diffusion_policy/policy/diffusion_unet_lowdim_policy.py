from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

class DiffusionUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: ConditionalUnet1D,
            noise_scheduler: DDPMScheduler,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            # guide-specific auxiliary loss (optional)
            collision_loss_weight: float = 0.0,
            collision_loss_margin: float = 0.0,
            collision_loss_robot_radius: float = 0.3,
            guide_action_mode: str = "forward_heading",
            guide_n_lookahead: int = 0,
            guide_n_obstacle_circles: int = 0,
            guide_n_obstacle_segments: int = 0,
            guide_obstacle_include_radius: bool = True,
            guide_segment_repr: str = "endpoints",
            # parameters passed to step
            **kwargs):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs

        # collision avoidance regularizer (only used in compute_loss)
        self.collision_loss_weight = float(collision_loss_weight)
        self.collision_loss_margin = float(collision_loss_margin)
        self.collision_loss_robot_radius = float(collision_loss_robot_radius)
        self.guide_action_mode = str(guide_action_mode).lower()
        self.guide_n_lookahead = int(guide_n_lookahead)
        self.guide_n_obstacle_circles = int(guide_n_obstacle_circles)
        self.guide_n_obstacle_segments = int(guide_n_obstacle_segments)
        self.guide_obstacle_include_radius = bool(guide_obstacle_include_radius)
        self.guide_segment_repr = str(guide_segment_repr).lower()
        if self.guide_segment_repr not in ("endpoints", "closest_dir"):
            raise ValueError(f"Unsupported guide_segment_repr: {self.guide_segment_repr}")
        self.last_base_loss = None
        self.last_collision_loss = None
        self.last_total_loss = None

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:,self.n_obs_steps:,:] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:,:self.n_obs_steps,:].reshape(
                obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        base_loss = F.mse_loss(pred, target, reduction='none')
        base_loss = base_loss * loss_mask.type(base_loss.dtype)
        base_loss = reduce(base_loss, 'b ... -> b (...)', 'mean')
        base_loss = base_loss.mean()

        collision_loss = None
        if (
            self.collision_loss_weight > 0
            and self.guide_action_mode == "forward_heading"
            and self.obs_as_global_cond
            and ("obs" in batch)
            and (self.guide_n_obstacle_circles > 0 or self.guide_n_obstacle_segments > 0)
        ):
            # Predict x0 from x_t and predicted noise.
            if pred_type == "epsilon":
                alphas_cumprod = self.noise_scheduler.alphas_cumprod
                if not torch.is_tensor(alphas_cumprod):
                    alphas_cumprod = torch.tensor(alphas_cumprod)
                alpha_bar = alphas_cumprod.to(
                    device=trajectory.device, dtype=trajectory.dtype
                )[timesteps]
                while alpha_bar.ndim < noisy_trajectory.ndim:
                    alpha_bar = alpha_bar.view(-1, *([1] * (noisy_trajectory.ndim - 1)))
                sqrt_alpha_bar = torch.sqrt(alpha_bar)
                sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)
                x0_pred = (noisy_trajectory - sqrt_one_minus_alpha_bar * pred) / torch.clamp(
                    sqrt_alpha_bar, min=1e-6
                )
            else:
                x0_pred = pred

            # enforce conditioning (if any)
            if condition_mask is not None:
                x0_pred = torch.where(condition_mask, trajectory, x0_pred)

            # unnormalize predicted actions to physical units
            action_pred = self.normalizer["action"].unnormalize(x0_pred)

            # slice the action window that will be executed at inference time
            To = self.n_obs_steps
            start = To
            if self.oa_step_convention:
                start = To - 1
            if self.pred_action_steps_only:
                action_slice = action_pred
            else:
                end = start + self.n_action_steps
                action_slice = action_pred[:, start:end]

            # parse obstacle features from the current observation step (robot-centric frame)
            obs = batch["obs"]
            obs0 = obs[:, start]
            base_offset = 4 + 2 * self.guide_n_lookahead
            circle_dim = 3 if self.guide_obstacle_include_radius else 2
            circles_total = self.guide_n_obstacle_circles * circle_dim
            seg_total = self.guide_n_obstacle_segments * 4

            circles = None
            segments = None
            if circles_total > 0:
                circles = obs0[:, base_offset : base_offset + circles_total].reshape(
                    obs0.shape[0], self.guide_n_obstacle_circles, circle_dim
                )
            if seg_total > 0:
                seg_offset = base_offset + circles_total
                segments = obs0[:, seg_offset : seg_offset + seg_total].reshape(
                    obs0.shape[0], self.guide_n_obstacle_segments, 4
                )

            # rollout kinematics in the observation frame: x forward, heading=0 at start
            B, N, _Da = action_slice.shape
            heading = torch.zeros((B,), device=action_slice.device, dtype=action_slice.dtype)
            pos = torch.zeros((B, 2), device=action_slice.device, dtype=action_slice.dtype)
            positions = []
            for i in range(N):
                forward = action_slice[:, i, 0]
                dtheta = action_slice[:, i, 1]
                heading = heading + dtheta
                step = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1) * forward.unsqueeze(-1)
                pos = pos + step
                positions.append(pos)
            pos_pred = torch.stack(positions, dim=1) if positions else pos.unsqueeze(1)

            margin = float(self.collision_loss_margin)
            robot_r = float(self.collision_loss_robot_radius)

            per_sample_pen = torch.zeros((B,), device=action_slice.device, dtype=action_slice.dtype)

            if circles is not None and circles.numel() > 0:
                centers = circles[..., :2]
                if self.guide_obstacle_include_radius:
                    radii = circles[..., 2]
                    valid = radii > 1e-6
                else:
                    radii = torch.zeros(
                        (B, centers.shape[1]), device=centers.device, dtype=centers.dtype
                    )
                    valid = torch.linalg.norm(centers, dim=-1) > 1e-6

                diff = pos_pred[:, :, None, :] - centers[:, None, :, :]
                dist = torch.linalg.norm(diff, dim=-1)
                dist = torch.where(valid[:, None, :], dist, torch.full_like(dist, 1e6))
                clearance = dist - (radii[:, None, :] + robot_r)
                circle_pen = torch.relu(margin - clearance) ** 2
                per_sample_pen = per_sample_pen + circle_pen.min(dim=-1).values.mean(dim=-1)

            if segments is not None and segments.numel() > 0:
                if self.guide_segment_repr == "endpoints":
                    p1 = segments[..., :2]
                    p2 = segments[..., 2:4]
                    ab = p2 - p1
                    denom = (ab * ab).sum(dim=-1, keepdim=True)
                    valid = denom.squeeze(-1) > 1e-8

                    ap = pos_pred[:, :, None, :] - p1[:, None, :, :]
                    t_proj = (ap * ab[:, None, :, :]).sum(dim=-1, keepdim=True) / torch.clamp(
                        denom[:, None, :, :], min=1e-8
                    )
                    t_proj = torch.clamp(t_proj, 0.0, 1.0)
                    closest = p1[:, None, :, :] + t_proj * ab[:, None, :, :]
                    diff = pos_pred[:, :, None, :] - closest
                    dist = torch.linalg.norm(diff, dim=-1)
                    dist = torch.where(valid[:, None, :], dist, torch.full_like(dist, 1e6))
                else:  # closest_dir, approximate walls as infinite lines
                    c = segments[..., :2]
                    d = segments[..., 2:4]
                    d_norm = torch.linalg.norm(d, dim=-1, keepdim=True)
                    valid = d_norm.squeeze(-1) > 1e-6
                    d_unit = d / torch.clamp(d_norm, min=1e-6)
                    delta = pos_pred[:, :, None, :] - c[:, None, :, :]
                    cross = delta[..., 0] * d_unit[:, None, :, 1] - delta[..., 1] * d_unit[:, None, :, 0]
                    dist = cross.abs()
                    dist = torch.where(valid[:, None, :], dist, torch.full_like(dist, 1e6))

                clearance = dist - robot_r
                seg_pen = torch.relu(margin - clearance) ** 2
                per_sample_pen = per_sample_pen + seg_pen.min(dim=-1).values.mean(dim=-1)

            # downweight collision penalty when the diffusion timestep is very noisy
            if pred_type == "epsilon":
                w = alpha_bar.reshape(B)
            else:
                w = torch.ones((B,), device=per_sample_pen.device, dtype=per_sample_pen.dtype)
            collision_loss = (per_sample_pen * w).mean()

        total_loss = base_loss
        if collision_loss is not None:
            total_loss = total_loss + (self.collision_loss_weight * collision_loss)

        self.last_base_loss = base_loss.detach()
        self.last_collision_loss = None if collision_loss is None else collision_loss.detach()
        self.last_total_loss = total_loss.detach()
        return total_loss
