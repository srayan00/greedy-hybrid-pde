import torch
import os
import numpy as np
import argparse
from ml_solver import MLSolver, DeepONet, FNOforPDE
from data_generation import GaussianRandomField, PDEDataset
from pde_pytorch import PoissonEquation1D, PoissonEquation2D

from trainer import Trainer, EarlyStopping, MSEalphaepsilonLoss
import json

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='deeponet', help='Model to use: deeponet or fno')
parser.add_argument('--dim', type=int, default=1, help='Dimension of the PDE: 1 or 2')
parser.add_argument("--boundary", type=str, default="Periodic", help="Boundary condition: Dirichlet or Periodic")
parser.add_argument("--in_channels", type=int, default=1, help="Number of input channels")
parser.add_argument("--extra", type=int, default=200, help="Extra data samples to generate beyond n_train + n_val")
parser.add_argument("--equation", type=str, default="Poisson", help="PDE to solve: Poisson")
parser.add_argument("--ckp_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
parser.add_argument("--model_name", type=str, default="model.pt", help="Model checkpoint name")
parser.add_argument("--data_dir", type=str, default="./data", help="Directory to save/load data")


if __name__ == "__main__":
    print("Parsing arguments...")
    args, unknown = parser.parse_known_args()
    model_type = args.model
    dim = args.dim
    boundary = args.boundary
    equation = args.equation
    ckp_dir = args.ckp_dir
    model_name = args.model_name
    data_dir = args.data_dir
    extra = args.extra
    in_channels = args.in_channels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if boundary not in ["Periodic", "Dirichlet"]:
        raise ValueError("Boundary condition must be either 'Dirichlet' or 'Periodic'")
    if equation not in ["Poisson"]:
        raise ValueError("Currently only Poisson equation is supported")
    if model_type not in ["deeponet", "fno"]:
        raise ValueError("Model must be either 'deeponet' or 'fno'")
    if dim not in [1, 2]:
        raise ValueError("Dimension must be either 1 or 2")
    if in_channels not in [1, 2]:
        raise ValueError("in_channels must be either 1 or 2")
    

    ckp_path = ckp_dir + f"/{model_type}_{model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_full.pth"
    save_path = ckp_dir + f"/{model_type}_{model_name}_{equation}_{boundary}_{dim}d_{in_channels}c"
    args_path = ckp_dir + f"/{model_type}_{model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_args.json"

    if os.path.exists(args_path):
        print(f"Loading training arguments from {args_path}...")
        with open(args_path, "r") as f:
            arguments = json.load(f)
    else:
        with open(f"args/{model_type}_args.json", "r") as f:
            arguments = json.load(f)
        with open(f"{args_path}", "w") as f:
            json.dump(arguments, f)
    
    # Creating/Loading Data
    print("Creating Data")
    train_data = []
    val_data = []

    if os.path.exists(f"{data_dir}/train_data_{equation}_{boundary}_{dim}d_{in_channels}c_{arguments["n_train"]}s.pt") and os.path.exists(f"{data_dir}/val_data_{equation}_{boundary}_{dim}d_{in_channels}c_{arguments["n_val"]}s.pt"):
        print(f"Loading data from {data_dir}...")
        with open(f"{data_dir}/train_data_{equation}_{boundary}_{dim}d_{in_channels}c_{arguments["n_train"]}s.pt", "rb") as f:
            train_data = torch.load(f)
        with open(f"{data_dir}/val_data_{equation}_{boundary}_{dim}d_{in_channels}c_{arguments["n_val"]}s.pt", "rb") as f:
            val_data = torch.load(f)
    print(f"This is what i loaded {len(train_data)}, {len(val_data)}")
    if len(train_data) < arguments["n_train"] or len(val_data) < arguments["n_val"]:
        with open(f"args/grf_args.json", "r") as f:
            arguments_grf = json.load(f)

        grf = GaussianRandomField(num_samples=arguments["N"],
                                    dim=dim,
                                    alpha=arguments_grf["alpha"],
                                    beta=arguments_grf["beta"],
                                    gamma=arguments_grf["gamma"],
                                    device=device,
                                    seed=1234)
        # if dim == 1:
        #     f = lambda x: np.sin(2 * np.pi * x)
        # else:
        #     f = lambda x, y: np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
        pushforward = None if boundary == "Dirichlet" else lambda x: x - torch.mean(x)
        f = grf.generate(arguments["n_train"] + arguments["n_val"] + extra, pushfoward=pushforward)
        if in_channels > 1:
            a = grf.generate(arguments["n_train"] + arguments["n_val"] + extra)
        else:
            if dim == 1:
                a = lambda x: 1.0
            else:
                a = lambda x, y: 1.0
        if boundary == "Dirichlet":
            x = torch.linspace(0, 1, arguments["N"], device=device, dtype=torch.float32)
            y = torch.linspace(0, 1, arguments["N"], device=device, dtype=torch.float32) if dim ==2 else None
        else:
            x = torch.linspace(0, 1, arguments["N"] + 1, device=device, dtype=torch.float32)[:-1]
            y = torch.linspace(0, 1, arguments["N"] + 1, device=device, dtype=torch.float32)[:-1] if dim ==2 else None
        
        start = len(train_data) + len(val_data)
        for i in range(start, arguments["n_train"] + arguments["n_val"] + extra):
            pde = None
            u_sol = None
            if dim == 1:
                pde = PoissonEquation1D(a_func=a[i] if in_channels > 1 else a,
                                        f_func=f[i],
                                        boundary=boundary,
                                        x=x, 
                                        device=device)
                u_sol = torch.tensor(pde.u, dtype=torch.float32, device=device)
                u_sol = u_sol - torch.mean(u_sol)
            else:
                pde = PoissonEquation2D(a_func=a[i].flatten() if in_channels > 1 else a,
                                        f_func=f[i].flatten(),
                                        boundary=boundary,
                                        x=x,
                                        y=y,
                                        device=device)
                new_shape = (arguments["N"], arguments["N"]) 
                u_sol = torch.tensor(pde.u.reshape(new_shape), dtype=torch.float32, device=device)
                u_sol = u_sol - torch.mean(u_sol)
                print(f"Generated sample {i+1}/{arguments['n_train'] + arguments['n_val'] + extra}")
                print(f"len(train_data) = {len(train_data)}, len(val_data) = {len(val_data)}")
            
            if in_channels > 1:
                input = torch.concatenate((a[i, None, :], f[i, None, :]), dim=0)
            else:
                input = f[i, None, :]
            residual = pde.compute_residual(u_sol.flatten())
            if torch.linalg.norm(residual) > 1:
                continue
            if len(train_data) < arguments["n_train"]:
                train_data.append((input, u_sol))
            else:
                val_data.append((input, u_sol))
            if len(train_data) == arguments["n_train"] and len(val_data) == arguments["n_val"]:
                break
            if i % 100 == 0:
                print(f"SAVE {i} samples")
                with open(f"{data_dir}/train_data_{equation}_{boundary}_{dim}d_{in_channels}c_{arguments["n_train"]}s.pt", "wb") as file:
                    torch.save(train_data, file)
                with open(f"{data_dir}/val_data_{equation}_{boundary}_{dim}d_{in_channels}c_{arguments["n_val"]}s.pt", "wb") as file:
                    torch.save(val_data, file)
                
        if len(train_data) < arguments["n_train"] or len(val_data) < arguments["n_val"]:
            print(f"Generated {len(train_data)} training samples and {len(val_data)} validation samples.")
            raise ValueError("Not enough data generated. Try increasing the extra variable.")
        with open(f"{data_dir}/train_data_{equation}_{boundary}_{dim}d_{in_channels}c_{arguments["n_train"]}s.pt", "wb") as f:
            torch.save(train_data, f)
        with open(f"{data_dir}/val_data_{equation}_{boundary}_{dim}d_{in_channels}c_{arguments["n_val"]}s.pt", "wb") as f:
            torch.save(val_data, f)
    print("Data creation/loading completed.")
    print(f"Train data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")
    print(f"Size of each input: {train_data[0][0].shape}, Size of each solution: {train_data[0][1].shape}")
    # Change this later 
    train_dataset = PDEDataset(train_data)
    val_dataset = PDEDataset(val_data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=arguments["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=arguments["batch_size"], shuffle=True)
    # train_dataset = PDEDataset(train_data[:2])
    # val_dataset = PDEDataset(train_data[:2])
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=False)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    print("Creating model...")
    if model_type == "deeponet":
        model = DeepONet(N=arguments["N"], dim=dim, in_channels=in_channels, device=device, boundary=boundary,
                        branch_dim=arguments["branch_dim"],
                        hidden_branch=arguments["hidden_branch"],
                        num_branch_layers=arguments["num_branch_layers"],
                        hidden_trunk=arguments["hidden_trunk"],
                        num_trunk_layers=arguments["num_trunk_layers"]).to(device)
    elif model_type == "fno":
        model = FNOforPDE(trunc_mode=arguments["trunc_mode"], dim=dim, in_channels=in_channels,
                          hidden_size=arguments["hidden_size"], num_layers=arguments["num_layers"]).to(device)
    ckp = None
    if os.path.exists(ckp_path):
        print(f"Loading model checkpoint from {ckp_path}...")
        ckp = torch.load(ckp_path, map_location=device)
    
    if ckp:
        print(f"Resuming training from epoch {ckp['epoch']}")
        model.load_state_dict(ckp["model"])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments["learning_rate"], weight_decay=arguments["weight_decay"])

    if ckp:
        optimizer.load_state_dict(ckp["optimizer"])
    
    scaler = torch.cuda.amp.GradScaler() 
    if ckp:
        scaler.load_state_dict(ckp["scaler"])
        print("AMP loaded")
    
    early_stopper = EarlyStopping(patience=arguments["patience"], verbose=True, delta=arguments["min_delta"], warmup_epochs=arguments["warmup_epochs_es"])

    if ckp:
        early_stopper.load_state_dict(ckp["early_stopping"])
        print("Early stopping state loaded")
    
    warm_up = lambda epoch: epoch / arguments["warmup_epochs_es"] if epoch <= arguments["warmup_epochs_es"] else 1
    # scheduler_wu = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_up)
    scheduler_wu = None

    # Load a learning rate scheduler if it exists
    if ckp is not None:
        if ckp["scheduler_wu"] is not None:
            scheduler_wu.load_state_dict(ckp["scheduler_wu"])
            print("Learning rates scheduler loaded", flush=True)

    # scheduler_re = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.1, patience=6, verbose=True)
    scheduler_re = None
    # Load a learning rate scheduler if it exists
    if ckp is not None:
        if ckp["scheduler_re"] is not None:
            scheduler_re.load_state_dict(ckp["scheduler_re"])
            print("Learning rates scheduler loaded", flush=True)

    scheduler_step = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=(arguments["epochs"] // 2), gamma=0.5)
    if ckp is not None:
        if ckp["scheduler_step"] is not None:
            scheduler_step.load_state_dict(ckp["scheduler_step"])

    loss_fn = MSEalphaepsilonLoss() # torch.nn.MSELoss()

    start_epoch = 0 if ckp is None else ckp["epoch"] + 1
    print("Starting training...")
    trainer = Trainer(model=model,
                      train_data=train_loader,
                      val_data=val_loader,
                      optimizer=optimizer,
                      device=device,
                      loss_fn=loss_fn,
                      save_every=1,
                      save_path=save_path,
                      parallel=False,
                      use_amp=True,
                      scheduled_sampler=None,
                      max_norm=arguments["max_norm"],
                      early_stopper=early_stopper,
                      warmup_epochs=arguments["warmup_epochs_es"],
                      lr_scheduler=[scheduler_wu, scheduler_re, scheduler_step])
    if ckp:
        print("loading losses")
        train_loss = ckp["train_losses"]
        val_loss = ckp["val_losses"]
    else:
        train_loss = None
        val_loss = None
    trainer.train(max_epochs=arguments["epochs"],
                  start_epoch=start_epoch,
                  train_losses=train_loss,
                  val_losses=val_loss)