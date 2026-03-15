import torch
import os
import numpy as np
import argparse
from ml_solver import DeepONet, FNOforPDE
from data_generation import GaussianRandomField, GaussianRandomFieldHierarchical, PDEDataset2
from pde import PoissonEquation1D, PoissonEquation2D, HelmholtzEquation1D, HelmholtzEquation2D, ConvectionDiffusion2D

from trainer import Trainer, EarlyStopping, MSEalphaepsilonLoss, RelativeMSELoss
import json

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='deeponet', help='Model to use: deeponet or fno')
parser.add_argument('--dim', type=int, default=1, help='Dimension of the PDE: 1 or 2')
parser.add_argument("--boundary", type=str, default="Periodic", help="Boundary condition: Dirichlet or Periodic")
parser.add_argument("--in_channels", type=int, default=1, help="Number of input channels")
parser.add_argument("--extra", type=int, default=200, help="Extra data samples to generate beyond n_train + n_val")
parser.add_argument("--equation", type=str, default="Poisson", help="PDE to solve: Poisson, Helmholtz, or ConvDiff")
parser.add_argument("--b_vel", type=float, default=20.0, help="Advection velocity magnitude for ConvDiff (b_vec = (b_vel, b_vel))")
parser.add_argument("--reaction_c", type=float, default=0.0, help="Reaction coefficient for ConvDiff (c in -div(a grad u) + b.grad u + c*u = f)")
parser.add_argument("--ckp_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
parser.add_argument("--model_name", type=str, default="model.pt", help="Model checkpoint name")
parser.add_argument('--data_name', type=str, default='', help='Name of the dataset to use (if not provided, a new dataset will be generated)')
parser.add_argument("--data_dir", type=str, default="./data", help="Directory to save/load data")
parser.add_argument("--grf_mode", type=str, default="fixed", choices=["fixed", "hierarchical"],
                    help="GRF mode: 'fixed' or 'hierarchical'")
parser.add_argument("--loss_alpha", type=float, default=None,
                    help="Override alpha for MSEalphaepsilonLoss (default: auto by dim/equation). Use -1.0 for RelativeMSELoss.")
parser.add_argument("--args_file", type=str, default=None,
                    help="Override default model args JSON file (e.g. args/deeponet_helmholtz_args.json)")
parser.add_argument("--k2_mode", type=str, default="exp", choices=["exp", "mild", "const"],
                    help="Helmholtz k2 pushforward: 'exp' (default), 'mild' (range ~[5,15]), 'const' (k2=10)")


if __name__ == "__main__":
    print("Parsing arguments...")
    args, unknown = parser.parse_known_args()
    model_type = args.model
    dim = args.dim
    boundary = args.boundary
    equation = args.equation
    ckp_dir = args.ckp_dir
    model_name = args.model_name
    data_name = args.data_name
    data_dir = args.data_dir
    extra = args.extra
    in_channels = args.in_channels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if boundary not in ["Periodic", "Dirichlet"]:
        raise ValueError("Boundary condition must be either 'Dirichlet' or 'Periodic'")
    if equation not in ["Poisson", "Helmholtz", "ConvDiff"]:
        raise ValueError("Currently only Poisson, Helmholtz, and ConvDiff are supported")
    if model_type not in ["deeponet", "fno"]:
        raise ValueError("Model must be either 'deeponet' or 'fno'")
    if dim not in [1, 2]:
        raise ValueError("Dimension must be either 1 or 2")
    if in_channels not in [1, 2]:
        raise ValueError("in_channels must be either 1 or 2")
    
    
    # Checkpoint and arguments setup
    ckp_path = ckp_dir + f"/{model_type}_{model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_full.pth"
    save_path = ckp_dir + f"/{model_type}_{model_name}_{equation}_{boundary}_{dim}d_{in_channels}c"
    args_path = ckp_dir + f"/{model_type}_{model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_args.json"

    # Loading arguments for the ML model
    if os.path.exists(args_path):
        print(f"Loading training arguments from {args_path}...")
        with open(args_path, "r") as f:
            arguments = json.load(f)
    else:
        default_args_file = args.args_file or f"args/{model_type}_args.json"
        print(f"Loading default arguments from {default_args_file}")
        with open(default_args_file, "r") as f:
            arguments = json.load(f)
        with open(f"{args_path}", "w") as f:
            json.dump(arguments, f)
    
    # Creating/Loading Data
    print("Creating Data")
    train_data = []
    val_data = []
    n_train = arguments["n_train"]
    n_val = arguments["n_val"]

    if os.path.exists(f"{data_dir}/{data_name}train_data_{equation}_{boundary}_{dim}d_{in_channels}c_{n_train}s.pt") and os.path.exists(f"{data_dir}/{data_name}val_data_{equation}_{boundary}_{dim}d_{in_channels}c_{n_val}s.pt"):
        print(f"Loading data from {data_dir}...")
        with open(f"{data_dir}/{data_name}train_data_{equation}_{boundary}_{dim}d_{in_channels}c_{n_train}s.pt", "rb") as f:
            train_data = torch.load(f)
        with open(f"{data_dir}/{data_name}val_data_{equation}_{boundary}_{dim}d_{in_channels}c_{n_val}s.pt", "rb") as f:
            val_data = torch.load(f)
        print(f"This is what i loaded {train_data[0].shape}, {val_data[0].shape}")
    if len(train_data) == 0 or len(val_data) == 0:
        with open(f"args/grf_args.json", "r") as f:
            arguments_grf = json.load(f)
        
        if args.grf_mode == "hierarchical":
            grf = GaussianRandomFieldHierarchical(num_samples=arguments["N"],
                                                  dim=dim,
                                                  alpha_min=0.01, alpha_max=100.0,
                                                  beta_min=0.1, beta_max=1000.0,
                                                  gamma_list=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
                                                  device=device, seed=1234)
        else:
            grf = GaussianRandomField(num_samples=arguments["N"],
                                        dim=dim,
                                        alpha=arguments_grf["alpha"],
                                        beta=arguments_grf["beta"],
                                        gamma=arguments_grf["gamma"],
                                        device=device,
                                        seed=1234)
        # Operator has a null space (constant mode) only for periodic Poisson
        # or periodic ConvDiff without reaction term.
        needs_mean_zero = equation == "Poisson" or (equation == "ConvDiff" and args.reaction_c == 0.0)

        if boundary == "Dirichlet":
            pushforward = None
        elif dim == 1:
            pushforward = lambda x: x - torch.mean(x, dim=-1, keepdim=True)
        else:
            pushforward = lambda x: x - torch.mean(x, dim=(-2, -1), keepdim=True)
        # Generate forcing functions
        if needs_mean_zero:
            f = grf.generate(n_train + n_val + extra, pushfoward=pushforward)
        else:
            f = grf.generate(n_train + n_val + extra, pushfoward=None)

        # For Helmholtz, normalize f to unit L2 norm per sample.
        # This reduces the solution magnitude range (PDE is linear in f).
        # At inference the residual is similarly normalized and output scaled back.
        if equation == "Helmholtz":
            f_flat = f.reshape(f.shape[0], -1)
            f_norms = torch.linalg.norm(f_flat, dim=-1, keepdim=True).clamp(min=1e-15)
            if dim == 1:
                f = f / f_norms
            else:
                f = f / f_norms.unsqueeze(-1)

        # Generate k2 function for Helmholtz
        if args.k2_mode == "const":
            k2_shape = (n_train + n_val + extra, arguments["N"], arguments["N"]) if dim == 2 else (n_train + n_val + extra, arguments["N"])
            k2 = 10.0 * torch.ones(k2_shape, device=device)
        elif args.k2_mode == "mild":
            k2 = grf.generate(n_train + n_val + extra, pushfoward=lambda x: 10.0 + 5.0 * torch.tanh(x))
        else:
            k2 = grf.generate(n_train + n_val + extra)
        if in_channels > 1:
            # Generate coefficient a for variable coefficient PDEs
            a = grf.generate(n_train + n_val + extra)
        else:
            if dim == 1:
                a = lambda x: 1.0
            else:
                a = lambda x, y: 1.0

        # Generate the grid for solving the PDEs
        if boundary == "Dirichlet":
            x = torch.linspace(0, 1, arguments["N"], device=device, dtype=torch.float32)
            y = torch.linspace(0, 1, arguments["N"], device=device, dtype=torch.float32) if dim ==2 else None
        else:
            x = torch.linspace(0, 1, arguments["N"] + 1, device=device, dtype=torch.float32)[:-1]
            y = torch.linspace(0, 1, arguments["N"] + 1, device=device, dtype=torch.float32)[:-1] if dim ==2 else None
        
        start = len(train_data) + len(val_data)

        # Create the equation object according to the specified parameters (equation, boundary, dim, in_channels)
        pde = None
        u_sol = None
        if dim == 1:
            if equation == "Poisson":
                pde = PoissonEquation1D(a_func=a, 
                                        f_func=f, 
                                        boundary=boundary, 
                                        x=x, device=device)
            else:
                pde = HelmholtzEquation1D(a_func=a, f_func=f, k2=k2, boundary=boundary,x=x,device=device)
            u_sol = torch.tensor(pde.u, dtype=torch.float32, device=device)
            u_sol = u_sol - torch.mean(u_sol, dim = -1, keepdim=True) if equation == "Poisson" and boundary == "Periodic" and in_channels == 1 else u_sol # this feels wrong because if the boundary == "Dirichlet" then this would produce an incorrect solution
        else:
            N2 = arguments["N"] * arguments["N"]
            if equation == "Poisson":
                pde = PoissonEquation2D(a_func=a.reshape(-1, N2) if in_channels > 1 else a, 
                                        f_func=f.reshape(-1, N2),
                                        boundary=boundary, 
                                        x=x, y=y, device=device)
            elif equation == "ConvDiff":
                pde = ConvectionDiffusion2D(a_func=a.reshape(-1, N2) if in_channels > 1 else a,
                                            f_func=f.reshape(-1, N2),
                                            b_vec=(args.b_vel, args.b_vel),
                                            boundary=boundary,
                                            x=x, y=y, device=device,
                                            reaction=args.reaction_c)
            else:
                pde = HelmholtzEquation2D(a_func=a.reshape(-1, N2) if in_channels > 1 else a, 
                                          f_func=f.reshape(-1, N2), 
                                          k2=k2.reshape(-1, N2),
                                          boundary=boundary, x=x, y=y, device=device)
            u_sol = torch.tensor(pde.u, dtype=torch.float32, device=device)
            u_sol = u_sol - torch.mean(u_sol, dim=(-2,-1), keepdim=True) if needs_mean_zero and boundary == "Periodic" and in_channels == 1 else u_sol

        # Defining the input and output of the ML model
        if in_channels > 1:
            if equation in ("Poisson", "ConvDiff"):
                input = torch.concatenate((a[:, None, :], f[:, None, :]), dim=1)
            else:
                input = torch.concatenate((a[:, None, :], k2[:, None, :], f[:, None, :]), dim=1)
        else:
            if equation in ("Poisson", "ConvDiff"):
                input = f[:, None, :]
            else:
                input = torch.concatenate((k2[:, None, :], f[:, None, :]), dim=1)

        # Compiling the training and validation data and saving it to disk
        train_data = [input[:n_train], u_sol[:n_train]]
        val_data = [input[n_train:(n_train + n_val)], u_sol[n_train:(n_train + n_val)]]
        with open(f"{data_dir}/{data_name}train_data_{equation}_{boundary}_{dim}d_{in_channels}c_{n_train}s.pt", "wb") as f:
            torch.save(train_data, f)
        with open(f"{data_dir}/{data_name}val_data_{equation}_{boundary}_{dim}d_{in_channels}c_{n_val}s.pt", "wb") as f:
            torch.save(val_data, f)

    print("Data creation/loading completed.")
    print(f"Train data size: {train_data[0].shape[0]}, Val data size: {val_data[0].shape[0]}")
    print(f"Size of each input: {train_data[0][0].shape}, Size of each solution: {train_data[1][0].shape}")
    
    # Creating Dataloaders for training and validation
    train_dataset = PDEDataset2(train_data)
    val_dataset = PDEDataset2(val_data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=arguments["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=arguments["batch_size"], shuffle=True)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    print("Creating model...")
    new_in_channels = in_channels + 1 if equation == "Helmholtz" else in_channels
    if model_type == "deeponet":
        model = DeepONet(N=arguments["N"], dim=dim, in_channels=new_in_channels, device=device, boundary=boundary,
                        branch_dim=arguments["branch_dim"],
                        hidden_branch=arguments["hidden_branch"],
                        num_branch_layers=arguments["num_branch_layers"],
                        hidden_trunk=arguments["hidden_trunk"],
                        num_trunk_layers=arguments["num_trunk_layers"]).to(device)
    elif model_type == "fno":
        model = FNOforPDE(trunc_mode=arguments["trunc_mode"], dim=dim, in_channels=new_in_channels,
                          hidden_size=arguments["hidden_size"], num_layers=arguments["num_layers"]).to(device)
    ckp = None

    # Loading checkpoint if it exists
    if os.path.exists(ckp_path):
        print(f"Loading model checkpoint from {ckp_path}...")
        ckp = torch.load(ckp_path, map_location=device)
    
    if ckp:
        print(f"Resuming training from epoch {ckp['epoch']}")
        model.load_state_dict(ckp["model"])
    
    # Creating optimizer, early stopping scheduler, learning rate schedulers
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments["learning_rate"], weight_decay=arguments["weight_decay"])

    if ckp:
        optimizer.load_state_dict(ckp["optimizer"])
    
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
    if args.loss_alpha is not None:
        if args.loss_alpha == -1.0:
            print("Using RelativeMSELoss (sample-level relative MSE)")
            loss_fn = RelativeMSELoss()
        else:
            alpha = args.loss_alpha
            print(f"Using MSEalphaepsilonLoss with alpha={alpha}")
            loss_fn = MSEalphaepsilonLoss(alpha=alpha)
    else:
        if dim == 1 and equation == "Poisson":
            alpha = 0.0
        elif dim == 1 and equation == "Helmholtz":
            alpha = 1.0
        else:
            alpha = 2.0
        print(f"Using MSEalphaepsilonLoss with alpha={alpha}")
        loss_fn = MSEalphaepsilonLoss(alpha=alpha)

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