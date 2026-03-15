import torch
import os
import numpy as np
import argparse
from ml_solver import DeepONet, FNOforPDE
from data_generation import  GaussianRandomFieldHierarchical, PDEDataset2, GaussianRandomField
from pde import PoissonEquation1D, PoissonEquation2D, HelmholtzEquation1D, HelmholtzEquation2D, ConvectionDiffusion2D
from numerical_solver import WeightedJacobiSolver, MultigridSolver, GaussSeidelSolver
from hybrid_solver import LSTMGreedyRouter, HybridSolver

from trainer import Trainer, EarlyStopping, ApproxGreedyRouterLoss, ScheduledSampler, ScheduledBPTT
import json

parser = argparse.ArgumentParser()
parser.add_argument('--ml_model', type=str, default='deeponet', help='Model to use: deeponet or fno')
parser.add_argument('--numerical_solvers', type=str, default='jacobi', help='comma-separated list of numerical solvers. Ex: jacobi_1.3,mg_2,gs')
parser.add_argument("--model", type=str, default='lstm')
parser.add_argument('--dim', type=int, default=1, help='Dimension of the PDE: 1 or 2')
parser.add_argument("--boundary", type=str, default="Periodic", help="Boundary condition: Dirichlet or Periodic")
parser.add_argument("--in_channels", type=int, default=1, help="Number of input channels")
parser.add_argument("--extra", type=int, default=200, help="Extra data samples to generate beyond n_train + n_val")
parser.add_argument("--equation", type=str, default="Poisson", help="PDE to solve: Poisson, Helmholtz, or ConvDiff")
parser.add_argument("--b_vel", type=float, default=20.0, help="Advection velocity for ConvDiff (b_vec=(b_vel,b_vel))")
parser.add_argument("--reaction_c", type=float, default=0.0, help="Reaction coefficient for ConvDiff (c in -div(a grad u) + b.grad u + c*u = f)")
parser.add_argument("--ckp_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
parser.add_argument("--ml_model_name", type=str, default="test", help="ml_model checkpoint name")
parser.add_argument("--model_name", type=str, default="", help="Model checkpoint name")
parser.add_argument('--data_name', type=str, default='', help='Name of the dataset to use (if not provided, a new dataset will be generated)')
parser.add_argument("--data_dir", type=str, default="./data", help="Directory to save/load data")
parser.add_argument("--grf_mode", type=str, default="fixed", choices=["fixed", "hierarchical"],
                    help="GRF mode: 'fixed' or 'hierarchical'")
parser.add_argument("--args_file", type=str, default=None,
                    help="Override path to LSTM args JSON file")
parser.add_argument("--k2_mode", type=str, default="exp", choices=["exp", "mild", "const"],
                    help="Helmholtz k2 pushforward: 'exp', 'mild', or 'const'")


if __name__ == "__main__":
    print("Parsing arguments...")
    args, unknown = parser.parse_known_args()
    model_type = args.model
    ml_model_type = args.ml_model
    dim = args.dim
    boundary = args.boundary
    equation = args.equation
    ckp_dir = args.ckp_dir
    model_name = args.model_name
    ml_model_name = args.ml_model_name
    data_name = args.data_name
    numerical_solvers = args.numerical_solvers.split(",")
    num_solvers = len(numerical_solvers) + 1
    data_dir = args.data_dir
    extra = args.extra
    in_channels = args.in_channels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if boundary not in ["Periodic", "Dirichlet"]:
        raise ValueError("Boundary condition must be either 'Dirichlet' or 'Periodic'")
    if equation not in ["Poisson", "Helmholtz", "ConvDiff"]:
        raise ValueError("Currently only Poisson, Helmholtz, and ConvDiff are supported")
    if ml_model_type not in ["deeponet", "fno"]:
        raise ValueError("Model must be either 'deeponet' or 'fno'")
    if dim not in [1, 2]:
        raise ValueError("Dimension must be either 1 or 2")
    if in_channels not in [1, 2]:
        raise ValueError("in_channels must be either 1 or 2")
    if model_type != "lstm":
        raise ValueError("Model must be LSTM")

    
    

    ml_ckp_path = ckp_dir + f"/{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_best.pth"
    ml_args_path = ckp_dir + f"/{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_args.json"

    ckp_path = ckp_dir + f"/{model_type}router_{model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_{args.numerical_solvers}_full.pth"
    save_path = ckp_dir + f"/{model_type}router_{model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_{args.numerical_solvers}"
    args_path = ckp_dir + f"/{model_type}router_{model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_{args.numerical_solvers}args.json"

    if os.path.exists(args_path):
        print(f"Loading training arguments from {args_path}...")
        with open(args_path, "r") as f:
            arguments = json.load(f)
    elif args.args_file is not None:
        with open(args.args_file, "r") as f:
            arguments = json.load(f)
    else:
        with open(f"args/{model_type}_args.json", "r") as f:
            arguments = json.load(f)
        
    
    if os.path.exists(ml_args_path):
        with open(ml_args_path, "r") as f:
            ml_arguments = json.load(f)
    else:
        raise ValueError("Path Not found")
    if arguments["N"] != ml_arguments["N"]:
        raise ValueError("N in ml arguments must match N in router arguments")
    
    with open(f"{args_path}", "w") as f:
        json.dump(arguments, f)
    # Creating/Loading Data
    print("Creating Data")
    n_train = 512
    n_val = 32
    if os.path.exists(f"{data_dir}/{data_name}router_train_data_{equation}_{boundary}_{dim}d_{in_channels}c_{n_train}s.pt") and os.path.exists(f"{data_dir}/{data_name}router_val_data_{equation}_{boundary}_{dim}d_{in_channels}c_{n_val}s.pt"):
        print(f"Loading data from {data_dir}...")
        with open(f"{data_dir}/{data_name}router_train_data_{equation}_{boundary}_{dim}d_{in_channels}c_{n_train}s.pt", "rb") as f:
            train_data = torch.load(f)
        with open(f"{data_dir}/{data_name}router_val_data_{equation}_{boundary}_{dim}d_{in_channels}c_{n_val}s.pt", "rb") as f:
            val_data = torch.load(f)
    else:
        with open(f"args/grf_args.json", "r") as f:
            arguments_grf = json.load(f)
        
        if args.grf_mode == "hierarchical":
            grf = GaussianRandomFieldHierarchical(num_samples=arguments["N"],
                                                  dim=dim,
                                                  alpha_min=0.01, alpha_max=100.0,
                                                  beta_min=0.1, beta_max=1000.0,
                                                  gamma_list=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
                                                  device=device, seed=34)
        else:
            grf = GaussianRandomField(num_samples=arguments["N"],
                                      dim=dim,
                                      alpha=arguments_grf["alpha"],
                                      beta=arguments_grf["beta"],
                                      gamma=arguments_grf["gamma"],
                                      device=device, seed=34)
        needs_mean_zero = equation == "Poisson" or (equation == "ConvDiff" and args.reaction_c == 0.0)

        if boundary == "Dirichlet":
            pushforward = None
        elif dim == 1:
            pushforward = lambda x: x - torch.mean(x, dim=-1, keepdim=True)
        else:
            pushforward = lambda x: x - torch.mean(x, dim=(-2, -1), keepdim=True)
        if needs_mean_zero:
            f = grf.generate(n_train + n_val + extra, pushfoward=pushforward)
        else:
            f = grf.generate(n_train + n_val + extra, pushfoward=None)

        # For Helmholtz, normalize f to unit L2 norm per sample (matches DeepONet training)
        if equation == "Helmholtz":
            f_flat = f.reshape(f.shape[0], -1)
            f_norms = torch.linalg.norm(f_flat, dim=-1, keepdim=True).clamp(min=1e-15)
            if dim == 1:
                f = f / f_norms
            else:
                f = f / f_norms.unsqueeze(-1)

        if args.k2_mode == "const":
            k2_shape = (n_train + n_val + extra, arguments["N"], arguments["N"]) if dim == 2 else (n_train + n_val + extra, arguments["N"])
            k2 = 10.0 * torch.ones(k2_shape, device=device)
        elif args.k2_mode == "mild":
            k2 = grf.generate(n_train + n_val + extra, pushfoward=lambda x: 10.0 + 5.0 * torch.tanh(x))
        else:
            k2 = grf.generate(n_train + n_val + extra)
        if in_channels > 1:
            a = grf.generate(n_train + n_val + extra)
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
            u_sol = u_sol - torch.mean(u_sol, dim = -1, keepdim=True) if equation == "Poisson" and boundary == "Periodic" else u_sol
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
            u_sol = u_sol - torch.mean(u_sol, dim=(-2,-1), keepdim=True) if needs_mean_zero and boundary == "Periodic" else u_sol
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
        train_data = [input[:n_train], u_sol[:n_train]]
        val_data = [input[n_train:(n_train + n_val)], u_sol[n_train:(n_train + n_val)]]
        with open(f"{data_dir}/{data_name}router_train_data_{equation}_{boundary}_{dim}d_{in_channels}c_{n_train}s.pt", "wb") as f:
            torch.save(train_data, f)
        with open(f"{data_dir}/{data_name}router_val_data_{equation}_{boundary}_{dim}d_{in_channels}c_{n_val}s.pt", "wb") as f:
            torch.save(val_data, f)
    print("Data creation/loading completed.")
    print(f"Train data size: {train_data[0].shape[0]}, Val data size: {val_data[0].shape[0]}")
    print(f"Size of each input: {train_data[0][0].shape}, Size of each solution: {train_data[1][0].shape}")
    # Change this later 
   
    train_dataset = PDEDataset2(train_data)
    val_dataset = PDEDataset2(val_data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=arguments["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=arguments["batch_size"], shuffle=True)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    print("Creating model...")
    new_in_channels = in_channels + 1 if equation == "Helmholtz" else in_channels
    if ml_model_type == "deeponet":
        ml_model = DeepONet(N=ml_arguments["N"], dim=dim, in_channels=new_in_channels, device=device, boundary=boundary,
                        branch_dim=ml_arguments["branch_dim"],
                        hidden_branch=ml_arguments["hidden_branch"],
                        num_branch_layers=ml_arguments["num_branch_layers"],
                        hidden_trunk=ml_arguments["hidden_trunk"],
                        num_trunk_layers=ml_arguments["num_trunk_layers"]).to(device)
    elif ml_model_type == "fno":
        ml_model = FNOforPDE(trunc_mode=ml_arguments["trunc_mode"], dim=dim, in_channels=new_in_channels,
                          hidden_size=ml_arguments["hidden_size"], num_layers=ml_arguments["num_layers"]).to(device)
    
    
    ml_ckp = None
    if os.path.exists(ml_ckp_path):
        print(f"Loading ml model checkpoint from {ml_ckp_path}...")
        ml_ckp = torch.load(ml_ckp_path, map_location=device, weights_only=False)
    
    if ml_ckp:
        ml_model.load_state_dict(ml_ckp["model"])
    
    if model_type == "lstm":
        if dim == 1:
            router = LSTMGreedyRouter(None, ml_arguments["N"]*(new_in_channels + 1), arguments["hidden_dim"], arguments["num_layers"], num_solvers, arguments["dropout"]).to(device)
        else:
            router = LSTMGreedyRouter(None, ml_arguments["N"]*ml_arguments["N"]*(new_in_channels + 1), arguments["hidden_dim"], arguments["num_layers"], num_solvers, arguments["dropout"]).to(device)

    ckp = None
    if os.path.exists(ckp_path):
        print(f"Loading model checkpoint from {ckp_path}...")
        ckp = torch.load(ckp_path, map_location=device)
    
    if ckp:
        print(f"Resuming training from epoch {ckp['epoch']}")
        router.load_state_dict(ckp["model"])
    print("Building the Numerical Solvers")
    if boundary == "Dirichlet":
        x = torch.linspace(0, 1, arguments["N"], device=device, dtype=torch.float32)
        y = torch.linspace(0, 1, arguments["N"], device=device, dtype=torch.float32) if dim ==2 else None
    else:
        x = torch.linspace(0, 1, arguments["N"] + 1, device=device, dtype=torch.float32)[:-1]
        y = torch.linspace(0, 1, arguments["N"] + 1, device=device, dtype=torch.float32)[:-1] if dim ==2 else None
    pde = None
    if equation == "Poisson":
        if dim == 1:
            pde = PoissonEquation1D(a_func= lambda x: 1,
                                    f_func=lambda x: 1,
                                    boundary=boundary,
                                    x=x, 
                                    device=device, 
                                    solve = False)
        else:
            pde = PoissonEquation2D(a_func=lambda x, y: 1,
                                            f_func=lambda x,y: 1,
                                            boundary=boundary,
                                            x=x,
                                            y=y,
                                            device=device, 
                                            solve = False)
    elif equation == "ConvDiff":
        pde = ConvectionDiffusion2D(a_func=lambda x, y: 1,
                                    f_func=lambda x, y: 1,
                                    b_vec=(args.b_vel, args.b_vel),
                                    boundary=boundary,
                                    x=x, y=y,
                                    device=device,
                                    solve=False,
                                    reaction=args.reaction_c)
    else:
        if dim == 1:
            pde = HelmholtzEquation1D(a_func= lambda x: 1, f_func=lambda x: 1, k2=lambda x: 1, boundary=boundary, x=x, device=device, solve = False)
        else:
            pde = HelmholtzEquation2D(a_func=lambda x, y: 1,f_func=lambda x,y: 1, k2=lambda x,y: 1, boundary=boundary,x=x,y=y, device=device, solve = False)


    list_of_solvers = []
    for solver in numerical_solvers:
        split = solver.split("_")
        if len(split) > 2:
            raise ValueError("Invalid Numerical Solver")
        if split[0] == "jacobi":
            if len(split) > 1:
                weight = float(split[1])
            else:
                weight = 1
            list_of_solvers.append(WeightedJacobiSolver(pde, device, weight))
        elif split[0] == "gs":
            list_of_solvers.append(GaussSeidelSolver(pde, device))
        elif split[0] == "mg":
            if len(split) > 1:
                levels = int(split[1])
            else:
                levels = 2
            print(f"This is device {device}")
            list_of_solvers.append(MultigridSolver(pde, levels, device))
        else:
            raise ValueError("Invalid Numerical Solver")
    print(f"List of solvers: {list_of_solvers}")

    model = HybridSolver(N=arguments["N"], dim=dim, in_channels=in_channels, boundary=boundary, equation=pde,
                                    suite_solver=list_of_solvers+[ml_model], router=router, tol=1e-7, max_iters=arguments["max_iters"], threshold=0.1)

    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments["learning_rate"], weight_decay=arguments["weight_decay"])

    if ckp:
        optimizer.load_state_dict(ckp["optimizer"])
    
    scaler = torch.cuda.amp.GradScaler() 
    # if ckp:
    #     scaler.load_state_dict(ckp["scaler"])
    #     print("AMP loaded")
    
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
    
    # Create a Scheduled sampler
    scheduled_sampler = ScheduledSampler(starting_teacher_forcing_prob=arguments["starting_teacher_forcing_prob"], ending_teacher_forcing_prob=arguments["ending_teacher_forcing_prob"],
                                         decay=arguments["decay"], warmup_epochs=arguments["warmup_epochs_ss"], linear=False)
    if ckp is not None:
        if ckp["scheduled_sampler"] is not None:
            scheduled_sampler.load_state_dict(ckp["scheduled_sampler"])
            print("Scheduled Sampler loaded", flush=True)

    # Create a Scheduled BPTT
    scheduled_bptt = ScheduledBPTT(max_iters=arguments["max_iters"], starting_bptt=arguments["starting_bptt"], linear_growth=arguments["linear_growth"], freq=arguments["freq"], warmup_epochs=arguments["warmup_epochs_ss"], linear = arguments["linear"])

    if ckp is not None:
        if "scheduler_bptt" in ckp and ckp["scheduler_bptt"] is not None:
            scheduled_bptt.load_state_dict(ckp["scheduler_bptt"])
            print("Scheduled Sampler loaded", flush=True)

    loss_fn = ApproxGreedyRouterLoss(centered=(needs_mean_zero and boundary == "Periodic"), normalized=False)

    start_epoch = 0 if ckp is None else ckp["epoch"] + 1
    print("Starting training...")
    # exit()
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
                      scheduled_sampler=scheduled_sampler,
                      scheduled_bptt=scheduled_bptt,
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