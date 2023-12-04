import warnings
warnings.filterwarnings("ignore")

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.distributed as dist
import torch.nn as nn
import functools
#from echo.src.base_objective import BaseObjective

from collections import defaultdict
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import numpy as np
import subprocess
import torch.fft
import logging
import shutil
import random

import optuna
import wandb
import tqdm
import glob
import os
import gc
import sys
import yaml

from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed.checkpoint as DCP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap
)
from torch.distributed.fsdp import StateDictType, FullStateDictConfig, ShardedStateDictConfig, OptimStateDictConfig, ShardedOptimStateDictConfig

from torchvision import transforms
from wpsml.model import ViTEncoderDecoder
from wpsml.data import ERA5Dataset, ToTensor, Normalize 
import joblib

from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullOptimStateDictConfig


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def setup(rank, world_size, mode):
    logging.info(f"Running {mode.upper()} on rank {rank} with world_size {world_size}.")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def cycle(dl):
    while True:
        for data in dl:
            yield data


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True


def launch_pbs_jobs(config):
    script_path = Path(__file__).absolute()
    script = f"""
    #!/bin/bash -l
    #PBS -N {config['pbs_job_params']['script_name']}
    #PBS -l select={config['pbs_job_params']['select']}
    #PBS -l walltime={config['pbs_job_params']['walltime']}
    #PBS -A {config['pbs_job_params']['account']}
    #PBS -q {config['pbs_job_params']['queue']}
    #PBS -o {config['pbs_job_params']['out_path']}
    #PBS -e {config['pbs_job_params']['err_path']}
    
    source ~/.bashrc
    {config['pbs']['bash']}
    python {script_path} -c {config} -w {config['pbs_job_params']['num_workers']} 1> /dev/null
    """
    with open("launcher.sh", "w") as fid:
        fid.write(script)
    jobid = subprocess.Popen(
        "qsub launcher.sh",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).communicate()[0]
    jobid = jobid.decode("utf-8").strip("\n")
    print(jobid)
    os.remove("launcher.sh")
    
    
class SimpleModel(nn.Module):
    def __init__(self, color_dim, surface_dim):
        super(SimpleModel, self).__init__()

        # Shared layers
        self.conv = nn.Conv3d(color_dim, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

        # Color prediction head
        self.conv_transpose_color = nn.ConvTranspose3d(64, color_dim, kernel_size=3, stride=1, padding=1)
        self.loss_fn_color = nn.MSELoss()

        # Surface detail prediction head
        self.conv_transpose_surface = nn.ConvTranspose2d(1, surface_dim, kernel_size=3, stride=1, padding=1)  # Using ConvTranspose2d for 2D prediction
        self.loss_fn_surface = nn.MSELoss()

    def forward(self, x, x_surface, y, y_surface):
        # Process 3D input
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv_transpose_color(x)
        loss = self.loss_fn_color(x, y)

        # Process 2D input
        x_surface = self.conv_transpose_surface(x_surface)  # Add channel dimension
        x_surface = x_surface.squeeze(1)  # Remove channel dimension after prediction
        loss_surface = self.loss_fn_surface(x_surface, y_surface)

        return x, x_surface, loss+loss_surface


def trainer(rank, world_size, conf, trial=False):
    if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
        setup(rank, world_size, conf["trainer"]["mode"])
        distributed = True
    else:
        distributed = False
    
    # infer device id from rank
    
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}") if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(rank % torch.cuda.device_count())
    
    # Config settings
    
    seed = 1000
    save_loc = conf['save_loc']
    train_batch_size = conf['trainer']['train_batch_size']
    valid_batch_size = conf['trainer']['valid_batch_size']
    batches_per_epoch = conf['trainer']['batches_per_epoch']
    valid_batches_per_epoch = conf['trainer']['valid_batches_per_epoch']
    learning_rate = conf['trainer']['learning_rate']
    weight_decay = conf['trainer']['weight_decay']
    start_epoch = conf['trainer']['start_epoch']
    epochs = conf['trainer']['epochs']
    amp = conf['trainer']['amp']
    grad_accum_every = conf['trainer']['grad_accum_every']
    apply_grad_penalty = conf['trainer']['apply_grad_penalty']
    thread_workers = conf['trainer']['thread_workers']
    stopping_patience = conf['trainer']['stopping_patience']

    # Model conf settings 
    
    image_height = conf['model']['image_height']
    image_width = conf['model']['image_width']
    patch_height = conf['model']['patch_height']
    patch_width = conf['model']['patch_width']
    frames = conf['model']['frames']
    frame_patch_size = conf['model']['frame_patch_size']
    
    dim = conf['model']['dim']
    layers = conf['model']['layers']
    dim_head = conf['model']['dim_head']
    mlp_dim = conf['model']['mlp_dim']
    heads = conf['model']['heads']
    depth = conf['model']['depth']

    rk4_integration = conf['model']['rk4_integration']
    num_register_tokens = conf['model']['num_register_tokens'] if 'num_register_tokens' in conf['model'] else 0
    use_registers = conf['model']['use_registers'] if 'use_registers' in conf['model'] else False
    token_dropout = conf['model']['token_dropout'] if 'token_dropout' in conf['model'] else 0.0
    
    use_codebook = conf['model']['use_codebook'] if 'use_codebook' in conf['model'] else False
    vq_codebook_size = conf['model']['vq_codebook_size'] if 'vq_codebook_size' in conf['model'] else 128
    vq_decay = conf['model']['vq_decay'] if 'vq_decay' in conf['model'] else 0.1
    vq_commitment_weight = conf['model']['vq_commitment_weight'] if 'vq_commitment_weight' in conf['model'] else 1.0
    
    use_vgg = conf['model']['use_vgg'] if 'use_vgg' in conf['model'] else False
    
    use_visual_ssl = conf['model']['use_visual_ssl'] if 'use_visual_ssl' in conf['model'] else False
    visual_ssl_weight = conf['model']['visual_ssl_weight'] if 'visual_ssl_weight' in conf['model'] else 0.05
    
    use_spectral_loss = conf['model']['use_spectral_loss']
    spectral_wavenum_init = conf['model']['spectral_wavenum_init'] if 'spectral_wavenum_init' in conf['model'] else 20
    spectral_lambda_reg = conf['model']['spectral_lambda_reg'] if 'spectral_lambda_reg' in conf['model'] else 1.0
    
    l2_recon_loss = conf['model']['l2_recon_loss'] if 'l2_recon_loss' in conf['model'] else False
    use_hinge_loss = conf['model']['use_hinge_loss'] if 'use_hinge_loss' in conf['model'] else True

    # Data vars
    channels = len(conf["data"]["variables"])
    surface_channels = len(conf["data"]["surface_variables"])
    
    # datasets (zarr reader) 
    all_ERA_files = sorted(glob.glob(conf["data"]["save_loc"]))
    
    # Specify the years to be excluded -- remove later
    years_to_exclude = ['1979', '1980', '1981', '1985', '1994']
    all_ERA_files = [file for file in all_ERA_files if not any(year in file for year in years_to_exclude)]
    
    train_dataset = ERA5Dataset(
        filenames=all_ERA_files[2:len(all_ERA_files)-1],
        transform=transforms.Compose([
            Normalize(conf["data"]["mean_path"],conf["data"]["std_path"]),
            ToTensor(),
        ]),
    )
    
    valid_dataset = ERA5Dataset(
        filenames=all_ERA_files[1:2],
        transform=transforms.Compose([
            Normalize(conf["data"]["mean_path"],conf["data"]["std_path"]),
            ToTensor(),
        ]),
    )
    
    # setup the distributed sampler
    
    sampler_tr = DistributedSampler(train_dataset,
                             num_replicas=world_size,
                             rank=rank,
                             shuffle=True,  # May be True
                             seed=seed, 
                             drop_last=True)
    
    sampler_val = DistributedSampler(valid_dataset,
                             num_replicas=world_size,
                             rank=rank,
                             seed=seed, 
                             shuffle=False, 
                             drop_last=True)
    
    # setup the dataloder for this process
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                              batch_size=train_batch_size, 
                              shuffle=False, 
                              sampler=sampler_tr, 
                              pin_memory=True, 
                              num_workers=thread_workers,
                              drop_last=True)
    
    valid_loader = torch.utils.data.DataLoader(valid_dataset, 
                              batch_size=valid_batch_size, 
                              shuffle=False, 
                              sampler=sampler_val, 
                              pin_memory=True, 
                              num_workers=thread_workers,
                              drop_last=True)
    
    # cycle for later so we can accum grad dataloader
    
    dl = cycle(train_loader)
    valid_dl = cycle(valid_loader)

    # model 
    vae = ViTEncoderDecoder(
        image_height=image_height,
        patch_height=patch_height,
        image_width=image_width,
        patch_width=patch_width,
        frames=frames,
        frame_patch_size=frame_patch_size,
        dim=dim,
        channels=channels,
        surface_channels=surface_channels,
        depth=depth,
        heads=heads,
        dim_head=dim_head,
        mlp_dim=mlp_dim,
        rk4_integration=rk4_integration,
        num_register_tokens=num_register_tokens,
        use_registers=use_registers,
        token_dropout=token_dropout,
        use_codebook=use_codebook,
        vq_codebook_size=vq_codebook_size,
        vq_decay=vq_decay,
        vq_commitment_weight=vq_commitment_weight,
        use_vgg=use_vgg,
        use_visual_ssl=use_visual_ssl,
        visual_ssl_weight=visual_ssl_weight,
        use_spectral_loss=use_spectral_loss,
        spectral_wavenum_init=spectral_wavenum_init,
        spectral_lambda_reg=spectral_lambda_reg,
        l2_recon_loss=l2_recon_loss,
        use_hinge_loss=use_hinge_loss
    )

    num_params = sum(p.numel() for p in vae.parameters())
    if rank == 0:
        logging.info(f"Number of parameters in the model: {num_params}")
        
    #summary(vae, input_size=(channels, height, width))

    # have to send the module to the correct device first
    vae.to(device)
    #vae = torch.compile(vae)

    # will not check that the device is correct here
    if conf["trainer"]["mode"] == "fsdp":
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=1_000_000
        )

        model = FSDP(
            vae,
            use_orig_params=True,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=torch.distributed.fsdp.MixedPrecision(
                param_dtype=torch.float16, 
                reduce_dtype=torch.float16, 
                buffer_dtype=torch.float16, 
                cast_forward_inputs=True
            ),
            #sharding_strategy=ShardingStrategy.FULL_SHARD # Zero3. Zero2 = ShardingStrategy.SHARD_GRAD_OP
        )
        
    elif conf["trainer"]["mode"] == "ddp":
        model = DDP(vae, device_ids=[device], output_device=None)
    else:
        model = vae

    # Load an optimizer, gradient scaler, and learning rate scheduler, the optimizer must come after wrapping model using FSDP
    if start_epoch == 0: # Loaded after loading model weights when reloading
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if conf["trainer"]["mode"] == "fsdp":
        scaler = ShardedGradScaler(enabled=amp)
    else:
        scaler = GradScaler(enabled=amp)

    scheduler = ReduceLROnPlateau(
        optimizer,
        patience=0,
        min_lr=0.001 * learning_rate,
        verbose=True
    )

    # load optimizer and grad scaler states
    if start_epoch > 0:

        checkpoint = torch.load(f"{save_loc}/checkpoint.pt", map_location=device)
        
        if conf["trainer"]["mode"] == "fsdp":
            logging.info(f"Loading FSDP model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}")

            # wait for all workers to get the model loaded
            torch.distributed.barrier()

            # tell torch how we are loading the data in (e.g. sharded states)
            FSDP.set_state_dict_type(
                model,
                StateDictType.SHARDED_STATE_DICT,
            )
            # different from ``torch.load()``, DCP requires model state_dict prior to loading to get
            # the allocated storage and sharding information.
            state_dict = {
                "model_state_dict": model.state_dict(),
            }
            DCP.load_state_dict(
                state_dict=state_dict,
                storage_reader=DCP.FileSystemReader(os.path.join(save_loc, "checkpoint")),
            )
            model.load_state_dict(state_dict["model_state_dict"])
            
            # Load the optimizer here on all ranks
            # https://github.com/facebookresearch/fairscale/issues/1083
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            curr_opt_state_dict = checkpoint["optimizer_state_dict"]
            optim_shard_dict = model.get_shard_from_optim_state_dict(curr_opt_state_dict)
            # https://www.facebook.com/pytorch/videos/part-5-loading-and-saving-models-with-fsdp-full-state-dictionary/421104503278422/
            # says to use scatter_full_optim_state_dict
            optimizer.load_state_dict(optim_shard_dict)


        elif conf["trainer"]["mode"] == "ddp":
            logging.info(f"Loading DDP model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}")
            model.module.load_state_dict(checkpoint["model_state_dict"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        else:
            logging.info(f"Loading model, optimizer, grad scaler, and learning rate scheduler states from {save_loc}")
            model.load_state_dict(checkpoint["model_state_dict"]) 
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Should not have to do this if we are doing this correctly ...
        # make sure the loaded optimizer is on the same device as the reloaded model
        # for state in optimizer.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.to(device)
        
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
            

    # Reload the results saved in the training csv if continuing to train
    if start_epoch == 0:
        results_dict = defaultdict(list)
    else:
        results_dict = defaultdict(list)
        saved_results = pd.read_csv(f"{save_loc}/training_log.csv")
        for key in saved_results.columns:
            if key == "index":
                continue
            results_dict[key] = list(saved_results[key])
        
    # Train 
    for epoch in range(start_epoch, epochs):

        train_loss = []

        # set up a custom tqdm
        batches_per_epoch = (
            batches_per_epoch if 0 < batches_per_epoch < len(train_loader) else len(train_loader)
        )

        batch_group_generator = tqdm.tqdm(
            range(batches_per_epoch), total=batches_per_epoch, leave=True
        )

        model.train()

        for steps in batch_group_generator:

            # logs

            logs = {}

            # update vae (generator)

            for _ in range(grad_accum_every):
                
                batch = next(dl)
                x = batch["x"].to(device)
                y1 = batch["y1"].to(device)
                x_surf = batch["x_surf"].to(device)
                y1_surf = batch["y1_surf"].to(device)

                with autocast(enabled=amp):
                    y1_pred, y1_pred_surf, loss = model(
                        x, x_surf, 
                        y1, y1_surf,
                        return_recons=True,
                        return_loss=True
                    )
                    
                    y2 = batch["y2"].to(device)
                    y2_surf = batch["y2_surf"].to(device)
                    
                    loss += model(
                        y1_pred.detach(), y1_pred_surf.detach(), 
                        y2, y2_surf,
                        return_recons=False,
                        return_loss=True
                    )

                    if use_visual_ssl:
                        loss += model(
                            x.detach(), x_surf.detach(), 
                            return_ssl_loss = True
                        )
                    
                    scaler.scale(loss / grad_accum_every).backward()

                accum_log(logs, {'loss': loss.item() / grad_accum_every})

            if distributed:
                torch.distributed.barrier()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            batch_loss = torch.Tensor([logs["loss"]]).cuda(device)
            if distributed:
                dist.all_reduce(batch_loss, dist.ReduceOp.AVG, async_op=False)
            train_loss.append(batch_loss[0].item())
            
            # update discriminator: not implemented yet
            
            to_print = "Epoch {} train_loss: {:.6f}".format(
                epoch, 
                np.mean(train_loss)
            )
            to_print += " lr: {:.12f}".format(optimizer.param_groups[0]["lr"])
            batch_group_generator.set_description(to_print)
            batch_group_generator.update(1)
            
        # Filter out non-float and NaN values from train_loss
        train_loss = [v for v in train_loss if isinstance(v, float) and np.isfinite(v)]

        # Shutdown the progbar
        batch_group_generator.close()
                
        if (not trial) and conf["trainer"]["mode"] != "fsdp": # rank == 0 and
                
            # Save the current model
    
            logging.info(f"Saving model, optimizer, grad scaler, and learning rate scheduler states to {save_loc}")

            state_dict = {
                "epoch": epoch,
                "model_state_dict": model.module.state_dict() if conf["trainer"]["mode"] == "ddp" else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }
            
            #with open(f"{save_loc}/checkpoint.pkl", "wb") as f:
            #    joblib.dump(state_dict, f)

            torch.save(state_dict, f"{save_loc}/checkpoint_{device}.pt" if conf["trainer"]["mode"] == "ddp" else f"{save_loc}/checkpoint.pt")

        elif not trial:

            logging.info(f"Saving FSDP model, optimizer, grad scaler, and learning rate scheduler states to {save_loc}")
            
            # https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
            FSDP.set_state_dict_type(
                model,
                StateDictType.SHARDED_STATE_DICT,
            )
            sharded_state_dict = {
                "model_state_dict": model.state_dict()
            }
            DCP.save_state_dict(
                state_dict=sharded_state_dict,
                storage_writer=DCP.FileSystemWriter(os.path.join(save_loc, "checkpoint")),
            )
            # save the optimizer
            optimizer_state = FSDP.full_optim_state_dict(model, optimizer)
            state_dict = {
                "epoch": epoch,
                "optimizer_state_dict": optimizer_state,
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict()
            }
            
            #with open(f"{save_loc}/checkpoint.pkl", "wb") as f:
            #    joblib.dump(state_dict, f)
        
            torch.save(state_dict, f"{save_loc}/checkpoint.pt")

        #if distributed:
        #    torch.distributed.barrier()
            
        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()

        ############
        #
        # Evaluation
        #
        ############
        
        model.eval()

        valid_loss = []

        # set up a custom tqdm
        
        valid_batches_per_epoch = (
            valid_batches_per_epoch if 0 < valid_batches_per_epoch < len(valid_loader) else len(valid_loader)
        )
        batch_group_generator = tqdm.tqdm(
            range(valid_batches_per_epoch), total=valid_batches_per_epoch, leave=True
        )

        with torch.no_grad():
            
            for k in batch_group_generator:
                
                batch = next(valid_dl)
                x = batch["x"].to(device)
                y1 = batch["y1"].to(device)
                x_surf = batch["x_surf"].to(device)
                y1_surf = batch["y1_surf"].to(device)
    
#                 y1_pred, y1_pred_surf, loss = model(
#                     x, x_surf, 
#                     y1, y1_surf,
#                     return_recons=True,
#                     return_loss=True,
#                     rk4_integration=True
#                 )
                
#                 del x, x_surf, y1, y1_surf
                
#                 y2 = batch["y2"].to(device)
#                 y2_surf = batch["y2_surf"].to(device)
                
#                 loss += model(
#                     y1_pred.detach(), y1_pred_surf.detach(), 
#                     y2, y2_surf,
#                     return_recons=False,
#                     return_loss=True,
#                     rk4_integration=True
#                 )

                y1_pred, y1_pred_surf = model(
                    x, x_surf, 
                    y1, y1_surf,
                    return_recons=True,
                    return_loss=False
                )
                
                y2 = batch["y2"].to(device)
                y2_surf = batch["y2_surf"].to(device)
                
                y2_pred, y2_pred_surf = model(
                    y1_pred.detach(), y1_pred_surf.detach(), 
                    y2, y2_surf,
                    return_recons=True,
                    return_loss=False
                )
                
                loss = nn.L1Loss()(y1, y1_pred) 
                loss += nn.L1Loss()(y1_surf, y1_pred_surf)
                loss += nn.L1Loss()(y2, y2_pred)
                loss += nn.L1Loss()(y2_surf, y2_pred_surf)
                
                batch_loss = torch.Tensor([loss.item()]).cuda(device)
                if distributed:
                    dist.all_reduce(batch_loss, dist.ReduceOp.AVG, async_op=False)
                valid_loss.append(batch_loss[0].item())

                # print to tqdm
                
                to_print = "Epoch {} valid_loss: {:.6f}".format(
                    epoch, np.mean(valid_loss)
                )
                batch_group_generator.set_description(to_print)
                batch_group_generator.update(1)

                if k >= valid_batches_per_epoch and k > 0:
                    break

                # Filter out non-float and NaN values from valid_loss
                valid_loss = [v for v in valid_loss if isinstance(v, float) and np.isfinite(v)]

        # Shutdown the progbar
        batch_group_generator.close()

        # Wait for rank-0 process to save the checkpoint above
        if distributed:
            torch.distributed.barrier()

        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()  

        if trial:
            if not np.isfinite(np.mean(valid_loss)):
                raise optuna.TrialPruned()  
      
        # Put things into a results dictionary -> dataframe
        results_dict["epoch"].append(epoch)
        results_dict["train_loss"].append(np.mean(train_loss))
        results_dict["valid_loss"].append(np.mean(valid_loss))
        results_dict["learning_rate"].append(optimizer.param_groups[0]["lr"])
        
        df = pd.DataFrame.from_dict(results_dict).reset_index()
        
        # Lower the learning rate if we are not improving
        scheduler.step(np.mean(valid_loss))

        # Save the best model so far
        if not trial:
            # save if this is the best model seen so far
            if (rank == 0) and (np.mean(valid_loss) == min(results_dict["valid_loss"])):
                if conf["trainer"]["mode"] == "ddp":
                    shutil.copy(f"{save_loc}/checkpoint_{device}.pt", f"{save_loc}/best_{device}.pt")
                elif conf["trainer"]["mode"] == "fsdp":
                    if os.path.exists(f"{save_loc}/best"):
                        shutil.rmtree(f"{save_loc}/best")
                    shutil.copytree(f"{save_loc}/checkpoint", f"{save_loc}/best")
                else:
                    shutil.copy(f"{save_loc}/checkpoint.pt", f"{save_loc}/best.pt")
                #shutil.copy(f"{save_loc}/checkpoint.pkl", f"{save_loc}/checkpoint.pkl")

        # Save the dataframe to disk
        if trial:
            df.to_csv(
                f"{save_loc}/trial_results/training_log_{trial.number}.csv",
                index=False,
            )
        else:
            df.to_csv(f"{save_loc}/training_log.csv", index=False)
            

        # Report result to the trial
        if trial:
            # Stop training if we have not improved after X epochs (stopping patience)
            best_epoch = [
                i
                for i, j in enumerate(results_dict["valid_loss"])
                if j == min(results_dict["valid_loss"])
            ][0]
            offset = epoch - best_epoch
            if offset >= stopping_patience:
                logging.info(f"Trial {trial.number} is stopping early")
                break

            if len(results_dict["valid_loss"]) == 0:
                raise optuna.TrialPruned()

    best_epoch = [
        i for i, j in enumerate(results_dict["valid_loss"]) if j == min(results_dict["valid_loss"])
    ][0]

    result = {k: v[best_epoch] for k,v in results_dict.items()}

    cleanup()

    return result


# class Objective(BaseObjective):
#     def __init__(self, config, metric="val_loss", device="cpu"):

#         # Initialize the base class
#         BaseObjective.__init__(self, config, metric, device)

#     def train(self, trial, conf):
#         try:
#             return trainer(conf, trial=trial)

#         except Exception as E:
#             if "CUDA" in str(E):
#                 logging.warning(
#                     f"Pruning trial {trial.number} due to CUDA memory overflow: {str(E)}."
#                 )
#                 raise optuna.TrialPruned()
#             elif "dilated" in str(E):
#                 raise optuna.TrialPruned()
#             else:
#                 logging.warning(f"Trial {trial.number} failed due to error: {str(E)}.")
#                 raise E



if __name__ == "__main__":

    description = "Train a segmengation model on a hologram data set"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        dest="model_config",
        type=str,
        default=False,
        help="Path to the model configuration (yml) containing your inputs.",
    )
    parser.add_argument(
        "-l",
        dest="launch",
        type=int,
        default=0,
        help="Submit {n_nodes} workers to PBS.",
    )
    parser.add_argument(
        "-w", 
        "--world-size", 
        type=int, 
        default=4, 
        help="Number of processes (world size) for multiprocessing"
    )
    args = parser.parse_args()
    args_dict = vars(args)
    config = args_dict.pop("model_config")
    launch = bool(int(args_dict.pop("launch")))

    # Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # Load the configuration and get the relevant variables
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    # Create directories if they do not exist and copy yml file
    os.makedirs(conf["save_loc"], exist_ok=True)
    if not os.path.exists(os.path.join(conf["save_loc"], "model.yml")):
        shutil.copy(config, os.path.join(conf["save_loc"], "model.yml"))

    # Launch PBS jobs
    if launch:
        logging.info("Launching to PBS")
        launch_pbs_jobs(config, conf["save_loc"])
        sys.exit()

#     wandb.init(
#         # set the wandb project where this run will be logged
#         project="Derecho parallelism",
#         name=f"Worker {os.environ["RANK"]} {os.environ["WORLD_SIZE"]}"
#         # track hyperparameters and run metadata
#         config={
#             "learning_rate": 0.5,
#             "architecture": "CNN",
#             "dataset": "CIFAR-10",
#             "epochs": 10,
#         }
#     )    

    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
        trainer(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), conf)
    else:
        trainer(0, 1, conf)
