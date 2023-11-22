from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.distributed as dist
import functools
#from echo.src.base_objective import BaseObjective

# from holodecml.seed import seed_everything
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
import psutil
import optuna
import wandb
import time
import tqdm
import os
import gc
import sys
import yaml
import warnings

from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

import xarray as xr
from torchvision import transforms
from wpsml.model import VQGanVAE
from wpsml.data import ERA5Dataset, ToTensor #get_forward_data, get_contiguous_segments, get_zarr_chunk_sequences,ERA5Dataset,ToTensor,worker_init_fn, ERA5Dataset2
#from typing import Optional, Callable, TypedDict, Union, Iterable, Tuple, NamedTuple, List
#from torchvision import transforms
#import pytorch_lightning as pl



warnings.filterwarnings("ignore")

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
    train_batch_size = conf['trainer']['train_batch_size']
    valid_batch_size = conf['trainer']['valid_batch_size']
    learning_rate = conf['trainer']['learning_rate']
    weight_decay = conf['trainer']['weight_decay']
    start_epoch = conf['trainer']['start_epoch']
    epochs = conf['trainer']['epochs']
    amp = conf['trainer']['amp']
    grad_accum_every = conf['trainer']['grad_accum_every']
    apply_grad_penalty = conf['trainer']['apply_grad_penalty']
    max_norm = conf['trainer']['grad_max_norm']

    # Model conf settings 
    image_height = conf['model']['image_height']
    image_width = conf['model']['image_width']
    patch_height = conf['model']['patch_height']
    patch_width = conf['model']['patch_width']
    frames = conf['model']['frames']
    frame_patch_size = conf['model']['frame_patch_size']
    
    channels = conf['model']['channels']
    dim = conf['model']['dim']
    layers = conf['model']['layers']
    dim_head = conf['model']['dim_head']
    mlp_dim = conf['model']['mlp_dim']
    heads = conf['model']['heads']
    depth = conf['model']['depth']
    
    vq_codebook_dim = conf['model']['vq_codebook_dim']
    vq_codebook_size = conf['model']['vq_codebook_size']
    vq_entropy_loss_weight = conf['model']['vq_entropy_loss_weight']
    vq_diversity_gamma = conf['model']['vq_diversity_gamma']
    discr_layers = conf['model']['discr_layers']
    
    # datasets (zarr reader)
    train_dataset = ERA5Dataset(
        filename=conf["data"]["save_loc"],
        transform=transforms.Compose([
            Normalize(conf["data"]["mean_path"],conf["data"]["std_path"]),
            ToTensor(),
        ]),
    )
    
    test_dataset = train_dataset
    
    # setup the distributed sampler
    sampler_tr = DistributedSampler(train_dataset,
                             num_replicas=world_size,
                             rank=rank,
                             shuffle=True,  # May be True
                             seed=seed, 
                             drop_last=True)
    
    sampler_val = DistributedSampler(test_dataset,
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
                              num_workers=0,
                              drop_last=True)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                              batch_size=valid_batch_size, 
                              shuffle=False, 
                              sampler=sampler_val, 
                              pin_memory=True, 
                              num_workers=0)
    
    # cycle for later so we can accum grad dataloader
    dl = cycle(train_loader)
    valid_dl = cycle(test_loader)
    
    # model 
    vae = VQGanVAE(
        image_height,
        patch_height,
        image_width,
        patch_width,
        frames,
        frame_patch_size,
        dim,
        channels,
        depth,
        heads,
        dim_head,
        mlp_dim,
        vq_codebook_dim=vq_codebook_dim,
        vq_codebook_size=vq_codebook_size,
        vq_entropy_loss_weight=vq_entropy_loss_weight,
        vq_diversity_gamma=vq_diversity_gamma,
        discr_layers=discr_layers
    ).to(device)

    num_params = sum(p.numel() for p in vae.parameters())
    print(f"Number of parameters in the model: {num_params}")

    # have to send the module to the correct device first
    vae.to(device)
    #vae = torch.compile(vae)
    
    # will not check that the device is correct here
    if conf["trainer"]["mode"] == "fsdp":
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=100_000 #1_000_000
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
            )
        )
    elif conf["trainer"]["mode"] == "ddp":
        model = DDP(vae, device_ids=[device], output_device=None)
    else:
        model = vae
    
    # Optimizer for ViT and for D
    all_parameters = set(model.parameters())
    discr_parameters = set(model.discr.parameters())
    vae_parameters = all_parameters - discr_parameters

    # adam with weight decay
    optimizer = torch.optim.AdamW(vae_parameters, lr=learning_rate, weight_decay=weight_decay)
    #discr_optim = torch.optim.AdamW(discr_parameters, lr=learning_rate, weight_decay=weight_decay)

    # grad scalers
    scaler = GradScaler(enabled=amp)
    #discr_scaler = GradScaler(enabled=amp)

    # Load a learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        patience=1,
        min_lr=1.0e-13,
        verbose=True
    )

    # Train 
    results_dict = defaultdict(list)
    
    for epoch in range(start_epoch, epochs):

        train_loss = []
        train_disc_loss = []

        # set up a custom tqdm
        batches_per_epoch = 100 #len(train_loader)
        batch_group_generator = tqdm.tqdm(
            range(batches_per_epoch), total=batches_per_epoch, leave=True
        )

        for steps in batch_group_generator:

            # update generator 
            model.train()

            # update vae (generator)

            # Step 1
            batch = next(dl)

            with autocast(enabled=amp):
                loss, y1_pred = model(
                    batch["x"].to(device),
                    batch["y1"].to(device),
                    return_loss=True,
                    return_recons=True,
                    add_gradient_penalty=apply_grad_penalty
                )
                scaler.scale(loss / grad_accum_every).backward()

            # clip grads
            torch.nn.utils.clip_grad_norm_(vae_parameters, max_norm=max_norm)
            
            if distributed:
                torch.distributed.barrier()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            batch_loss = torch.Tensor([loss.item()]).cuda(device)
            dist.all_reduce(batch_loss, dist.ReduceOp.AVG, async_op=False)
            results_dict["train_loss_0"].append(batch_loss[0].item())

            # Step 2
            with autocast(enabled=amp):
                loss = model(
                    y1_pred.detach(),
                    batch["y2"].to(device),
                    return_loss=True,
                    add_gradient_penalty=apply_grad_penalty
                )
                scaler.scale(loss / grad_accum_every).backward()

            # clip grads
            torch.nn.utils.clip_grad_norm_(vae_parameters, max_norm=max_norm)

            if distributed:
                torch.distributed.barrier()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            batch_loss = torch.Tensor([loss.item()]).cuda(device)
            dist.all_reduce(batch_loss, dist.ReduceOp.AVG, async_op=False)
            results_dict["train_loss_1"].append(batch_loss[0].item())
            
            # update discriminator

            # if model.discr is not None:
            #     for _ in range(grad_accum_every):
            #         batch = next(dl)
            #         x = batch["x"].to(device)
            #         y1 = batch["y1"].to(device)

            #         with autocast(enabled=amp):
            #             loss = model(x, y1, return_discr_loss=True)
            #             discr_scaler.scale(loss / grad_accum_every).backward()

            #         accum_log(logs, {'discr_loss': loss.item() / grad_accum_every})

            #     if distributed:
            #         torch.distributed.barrier()
            #     discr_scaler.step(discr_optim)
            #     discr_scaler.update()
            #     discr_optim.zero_grad()

            #     # log

            #     print(f"{steps}: vae loss: {logs['loss']} - discr loss: {logs['discr_loss']}")

            #     batch_disc_loss = torch.Tensor([logs["discr_loss"]]).cuda(device)
            #     dist.all_reduce(batch_disc_loss, dist.ReduceOp.AVG, async_op=False)
            #     results_dict["train_disc_loss"].append(batch_disc_loss[0])
            
            # update tqdm
            # to_print = "Epoch {} train_loss: {:.6f} disc_loss: {:.6f}".format(
            #     epoch, 
            #     np.mean(results_dict["train_loss"]), 
            #     np.mean(results_dict["train_disc_loss"])
            # )
            to_print = "Epoch {} train_loss_0: {:.6f} train_loss_1: {:.6f}".format(
                epoch, 
                np.mean(results_dict["train_loss_0"]),
                np.mean(results_dict["train_loss_1"])
            )
            to_print += " lr: {:.12f}".format(optimizer.param_groups[0]["lr"])
            batch_group_generator.set_description(to_print)
            batch_group_generator.update()

        # Shutdown the progbar
        batch_group_generator.close()

        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()

        # Test the model
        model.eval()
        with torch.no_grad():
            # set up a custom tqdm
            valid_batches_per_epoch = 10 #len(test_loader)
            batch_group_generator = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=True)

            valid_loss = []
            for k, batch in batch_group_generator:
                
                loss, y1_pred = model(
                    batch["x"].to(device),
                    batch["y1"].to(device),
                    return_loss=True,
                    return_recons=True
                )

                loss += model(
                    y1_pred.detach(),
                    y1,
                    return_loss=True
                )
                
                # update tqdm
                batch_loss = torch.Tensor([loss]).cuda(device)
                dist.all_reduce(batch_loss, dist.ReduceOp.AVG, async_op=False)
                valid_loss.append(batch_loss[0])
                to_print = "Epoch {} valid_loss: {:.6f}".format(
                    epoch, np.mean(valid_loss)
                )
                batch_group_generator.set_description(to_print)
                batch_group_generator.update()
                
                if k >= valid_batches_per_epoch and k > 0:
                    break
                    
            # Shutdown the progbar
            batch_group_generator.close()

        # clear the cached memory from the gpu
        torch.cuda.empty_cache()
        gc.collect()   
            
        if trial:
            if not np.isfinite(np.mean(valid_loss)):
                raise optuna.TrialPruned()  
                
        # Put things into a results dictionary -> dataframe
        results_dict["epoch"].append(epoch)
        results_dict["train_loss_0"].append(np.mean(results_dict["train_loss_0"]))
        results_dict["train_loss_1"].append(np.mean(results_dict["train_loss_1"]))
        #results_dict["train_disc_loss"].append(np.mean(train_disc_loss))
        results_dict["valid_loss"].append(np.mean(valid_loss))
        results_dict["learning_rate"].append(optimizer.param_groups[0]["lr"])
        df = pd.DataFrame.from_dict(results_dict).reset_index()

        # Save the model if its the best so far.
        if not trial and (np.mean(valid_loss) == min(results_dict["valid_loss"])):
            state_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": np.mean(valid_loss),
            }
            torch.save(state_dict, f"{save_loc}/best.pt")

        # Save the dataframe to disk
        if trial:
            df.to_csv(
                f"{save_loc}/trial_results/training_log_{trial.number}.csv",
                index=False,
            )
        else:
            df.to_csv(f"{save_loc}/training_log.csv", index=False)

        # Lower the learning rate if we are not improving
        lr_scheduler.step(np.mean(valid_loss))

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

    trainer(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), conf)
    #trainer(0, 1, conf)
