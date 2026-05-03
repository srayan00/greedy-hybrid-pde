import torch
import os
import numpy as np
import argparse
from ml_solver import DeepONet, FNOforPDE, DeepONetCNN
from data_generation import GaussianRandomField, GaussianRandomFieldHierarchical, PDEDataset2
from pde import PoissonEquation1D, PoissonEquation2D, HelmholtzEquation1D, HelmholtzEquation2D, ConvectionDiffusion2D
from numerical_solver import WeightedJacobiSolver

from trainer import Trainer, EarlyStopping, MSEalphaepsilonLoss
import json

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='deeponet', help='Model to use: deeponet/fno/deeponetcnn')
parser.add_argument('--dim', type=int, default=2, help='Dimension of the PDE: 1 or 2')
parser.add_argument("--boundary", type=str, default="Periodic", help="Boundary condition: Dirichlet or Periodic")
parser.add_argument("--in_channels", type=int, default=1, help="Number of input channels")
parser.add_argument("--extra", type=int, default=200, help="Extra data samples to generate beyond n_train + n_val")
parser.add_argument("--equation", type=str, default="Poisson", help="PDE to solve: Poisson")
parser.add_argument("--ckp_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
parser.add_argument("--model_name", type=str, default="model.pt", help="Model checkpoint name")
parser.add_argument('--data_name', type=str, default='', help='Name of the dataset to use (if not provided, a new dataset will be generated)')
parser.add_argument("--data_dir", type=str, default="./data", help="Directory to save/load data")
parser.add_argument("--grf_mode", type = str, default="fixed", help="Mode of the GRF: hierarchical or fixed")
parser.add_argument("--loss_alpha", type = float, default=0.0, help="Alpha for the loss function")
parser.add_argument("--b_vel", type=float, default=20.0, help="Advection velocity magnitude for ConvDiff (b_vec = (b_vel, b_vel))")
parser.add_argument("--reaction_c", type=float, default=0.0, help="Reaction coefficient for ConvDiff (c in -div(a grad u) + b.grad u + c*u = f)")



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
    grf_mode = args.grf_mode
    loss_alpha = args.loss_alpha
    b_vel = args.b_vel
    reaction_c = args.reaction_c

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if boundary not in ["Periodic", "Dirichlet"]:
        raise ValueError("Boundary condition must be either 'Dirichlet' or 'Periodic'")
    if equation not in ["Poisson", "Helmholtz", "ConvDiff"]:
        raise ValueError("Currently only Poisson, Helmholtz and ConvectionDiffusion equation are supported")
    if model_type not in ["deeponet", "fno", "deeponetcnn"]:
        raise ValueError("Model must be either 'deeponet' or 'fno' or 'deeponetcnn'")
    if dim not in [1, 2]:
        raise ValueError("Dimension must be either 1 or 2")
    if in_channels not in [1, 2]:
        raise ValueError("in_channels must be either 1 or 2")
    
    
    # Checkpoint and arguments setup
    ckp_path = ckp_dir + f"/{model_type}_{model_name}_{grf_mode}_{equation}_{boundary}_{dim}d_{in_channels}c_full.pth"
    save_path = ckp_dir + f"/{model_type}_{model_name}_{grf_mode}_{equation}_{boundary}_{dim}d_{in_channels}c"
    args_path = ckp_dir + f"/{model_type}_{model_name}_{grf_mode}_{equation}_{boundary}_{dim}d_{in_channels}c_args.json"

    # Loading arguments for the ML model
    if os.path.exists(args_path):
        print(f"Loading training arguments from {args_path}...")
        with open(args_path, "r") as f:
            arguments = json.load(f)
    else:
        if os.path.exists(f"args/{model_type}_{dim}d_args.json"):
            with open(f"args/{model_type}_{dim}d_args.json", "r") as f:
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
    n_train = arguments["n_train"]
    n_val = arguments["n_val"]

    if os.path.exists(f"{data_dir}/{data_name}train_data_{grf_mode}_{equation}_{boundary}_{dim}d_{in_channels}c_{n_train}s.pt") and os.path.exists(f"{data_dir}/{data_name}val_data_{grf_mode}_{equation}_{boundary}_{dim}d_{in_channels}c_{n_val}s.pt"):
        print(f"Loading data from {data_dir}...")
        with open(f"{data_dir}/{data_name}train_data_{grf_mode}_{equation}_{boundary}_{dim}d_{in_channels}c_{n_train}s.pt", "rb") as f:
            train_data = torch.load(f)
        with open(f"{data_dir}/{data_name}val_data_{grf_mode}_{equation}_{boundary}_{dim}d_{in_channels}c_{n_val}s.pt", "rb") as f:
            val_data = torch.load(f)
        print(f"This is what i loaded {train_data[0].shape}, {val_data[0].shape}")
    if len(train_data) == 0 or len(val_data) == 0:
        if grf_mode == "hierarchical":
            with open(f"args/hierarchical_grf.json", "r") as f:
                arguments_grf = json.load(f)
            grf = GaussianRandomFieldHierarchical(num_samples=arguments["N"],
                                                    dim=dim,
                                                    alpha_min=arguments_grf["alpha_min"],
                                                    alpha_max=arguments_grf["alpha_max"],
                                                    beta_min=arguments_grf["beta_min"],
                                                    beta_max=arguments_grf["beta_max"],
                                                    gamma_list=arguments_grf["gamma_list"],
                                                    device=device, seed=1234)
        else:
            with open(f"args/grf_args.json", "r") as f:
                arguments_grf = json.load(f)
            grf = GaussianRandomField(num_samples=arguments["N"],
                                        dim=dim,
                                        alpha=arguments_grf["alpha"],
                                        beta=arguments_grf["beta"],
                                        gamma=arguments_grf["gamma"],
                                        device=device,
                                        seed=1234)
        if dim == 1:
            pushforward = lambda x: x - torch.mean(x, dim=-1, keepdim=True)
        else:
            pushforward = lambda x: x - torch.mean(x, dim=(-2, -1), keepdim=True)
        needs_mean_zero = (equation == "Poisson") or (equation == "ConvDiff" and reaction_c == 0.0)
        # Generate forcing functions
        f = grf.generate(n_train + n_val + extra, pushfoward=pushforward) if needs_mean_zero and boundary == "Periodic" and in_channels == 1 else grf.generate(n_train + n_val + extra, pushfoward=None)

        # if equation == "Poisson" or (equation == "Helmholtz" and in_channels > 1):
        #     f = grf.generate(n_train + n_val + extra, pushfoward=pushforward) if equation == "Poisson" and in_channels == 1 else grf.generate(n_train + n_val + extra, pushfoward=None)
        # else:
        #     if dim == 1:
        #         f = lambda x: 0.0
        #     else:
        #         f = lambda x, y: 0.0

        # Generate k2 function for Helmholtz
        k2 = grf.generate(n_train + n_val + extra)
        # if (equation == "Poisson" and in_channels > 1) or (equation == "Helmholtz" and in_channels > 2):
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
                                        boundary=boundary, in_channels=in_channels,
                                        x=x, device=device)
            else:
                pde = HelmholtzEquation1D(a_func=a, f_func=f, k2=k2, boundary=boundary,x=x,device=device)
            u_sol = torch.tensor(pde.u, dtype=torch.float32, device=device)
            u_sol = u_sol - torch.mean(u_sol, dim = -1, keepdim=True) if equation == "Poisson" and boundary == "Periodic" and in_channels == 1 else u_sol # this feels wrong because if the boundary == "Dirichlet" then this would produce an incorrect solution
        else:
            if equation == "Poisson":
                pde = PoissonEquation2D(a_func=a.reshape(-1, arguments["N"] * arguments["N"]) if in_channels > 1 else a, 
                                        f_func=f.reshape(-1, arguments["N"] * arguments["N"]),
                                        boundary=boundary, in_channels=in_channels,
                                        x=x, y=y, device=device)
            elif equation == "ConvDiff":
                pde = ConvectionDiffusion2D(a_func=a.reshape(-1, arguments["N"] * arguments["N"]) if in_channels > 1 else a,
                                            f_func=f.reshape(-1, arguments["N"] * arguments["N"]),
                                            b_vec=(b_vel, b_vel),
                                            boundary=boundary,
                                            x=x, y=y, device=device,
                                            reaction=reaction_c)
            else:
                pde = HelmholtzEquation2D(a_func=a.reshape(-1, arguments["N"] * arguments["N"]) if in_channels > 1 else a, 
                                          f_func= f.reshape(-1, arguments["N"] * arguments["N"]), 
                                          k2=k2.reshape(-1, arguments["N"] * arguments["N"]),
                                          boundary=boundary, x=x, y=y, device=device)
            u_sol = torch.tensor(pde.u, dtype=torch.float32, device=device)
            u_sol = u_sol - torch.mean(u_sol, dim=(-2,-1), keepdim=True) if needs_mean_zero and boundary == "Periodic" and in_channels == 1 else u_sol

        # Defining the input and output of the ML model
        # if in_channels == 1:
        #     if equation == "Poisson":
        #         input = f[:, None, :]
        #     else:
        #         input = k2[:, None, :]
        # elif in_channels == 2:
        #     if equation == "Poisson":
        #         input = torch.concatenate((a[:, None, :], f[:, None, :]), dim=1)
        #     else:
        #         input = torch.concatenate((k2[:, None, :], f[:, None, :]), dim=1)
        # else:
        #     if equation == "Poisson":
        #         raise ValueError("Poisson with in_channels > 2 is not supported")
        #     else:
        #         input = torch.concatenate((a[:, None, :], k2[:, None, :], f[:, None, :]), dim=1)
        if in_channels > 1:
            if equation in ["Poisson", "ConvDiff"]:
                input = torch.concatenate((a[:, None, :], f[:, None, :]), dim=1)
            else:
                input = torch.concatenate((a[:, None, :], k2[:, None, :], f[:, None, :]), dim=1)
        else:
            if equation in ["Poisson", "ConvDiff"]:
                input = f[:, None, :]
            else:
                input = torch.concatenate((k2[:, None, :], f[:, None, :]), dim=1)

        # Compiling the training and validation data and saving it to disk
        train_data = [input[:n_train], u_sol[:n_train]]
        val_data = [input[n_train:(n_train + n_val)], u_sol[n_train:(n_train + n_val)]]
        with open(f"{data_dir}/{data_name}train_data_{grf_mode}_{equation}_{boundary}_{dim}d_{in_channels}c_{n_train}s.pt", "wb") as f:
            torch.save(train_data, f)
        with open(f"{data_dir}/{data_name}val_data_{grf_mode}_{equation}_{boundary}_{dim}d_{in_channels}c_{n_val}s.pt", "wb") as f:
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
        model = FNOforPDE(trunc_mode=arguments["trunc_mode"], dim=dim, N = arguments["N"], in_channels=new_in_channels,
                          hidden_size=arguments["hidden_size"], num_layers=arguments["num_layers"]).to(device)
    elif model_type == "deeponetcnn":
        model = DeepONetCNN(N=arguments["N"], dim=dim, in_channels=new_in_channels, device=device, boundary=boundary,
                        branch_dim=arguments["branch_dim"],
                        hidden_branch_channels=arguments["hidden_branch_channels"],
                        kernel_size=arguments["kernel_size"],
                        stride=arguments["stride"],
                        hidden_trunk=arguments["hidden_trunk"],
                        hidden_branch=arguments["hidden_branch"]).to(device)
    ckp = None

    # Loading checkpoint if it exists
    if os.path.exists(ckp_path):
        print(f"Loading model checkpoint from {ckp_path}...")
        ckp = torch.load(ckp_path, map_location=device, weights_only=False)
    
    if ckp:
        print(f"Resuming training from epoch {ckp['epoch']}")
        if model_type == "fno" and "_metadata" in ckp["model"]:
            del ckp["model"]["_metadata"]
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
    # if dim == 1 and equation == "Poisson":
    #     alpha = 0.0
    # elif dim == 1 and equation == "Helmholtz":
    #     alpha = 1.0
    # else:
    #     alpha = 2.0

    loss_fn = MSEalphaepsilonLoss(alpha=loss_alpha) # torch.nn.MSELoss()

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