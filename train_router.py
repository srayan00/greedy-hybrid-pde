import torch
import os
import numpy as np
import argparse
from ml_solver import MLSolver, DeepONet, FNOforPDE
from data_generation import GaussianRandomField, PDEDataset
from pde_pytorch import PoissonEquation1D, PoissonEquation2D, HelmholtzEquation1D, HelmholtzEquation2D
from numerical_solver_pytorch import WeightedJacobiSolver, MultigridSolver, GaussSeidelSolver
from hybrid_solver import Router, ConstantRouter, HINTSRouter, LSTMGreedyRouter, HybridSolver

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
parser.add_argument("--equation", type=str, default="Poisson", help="PDE to solve: Poisson")
parser.add_argument("--ckp_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
parser.add_argument("--ml_model_name", type=str, default="test", help="ml_model checkpoint name")
parser.add_argument("--model_name", type=str, default="", help="Model checkpoint name")
parser.add_argument("--data_dir", type=str, default="./data", help="Directory to save/load data")


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
    numerical_solvers = args.numerical_solvers.split(",")
    num_solvers = len(numerical_solvers) + 1
    data_dir = args.data_dir
    extra = args.extra
    in_channels = args.in_channels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if boundary not in ["Periodic", "Dirichlet"]:
        raise ValueError("Boundary condition must be either 'Dirichlet' or 'Periodic'")
    if equation not in ["Poisson", "Helmholtz"]:
        raise ValueError("Currently only Poisson/Helmholtz equation is supported")
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
    else:
        with open(f"args/{model_type}_args.json", "r") as f:
            arguments = json.load(f)
        with open(f"{args_path}", "w") as f:
            json.dump(arguments, f)
    
    if os.path.exists(ml_args_path):
        with open(ml_args_path, "r") as f:
            ml_arguments = json.load(f)
    else:
        raise ValueError("Path Not found")
    # Creating/Loading Data
    print("Creating Data")

    if os.path.exists(f"{data_dir}/router_train_data_{equation}_{boundary}_{dim}d_{in_channels}c_{arguments["n_train"]}s.pt") and os.path.exists(f"{data_dir}/router_val_data_{equation}_{boundary}_{dim}d_{in_channels}c_{arguments["n_val"]}s.pt"):
        print(f"Loading data from {data_dir}...")
        with open(f"{data_dir}/router_train_data_{equation}_{boundary}_{dim}d_{in_channels}c_{arguments["n_train"]}s.pt", "rb") as f:
            train_data = torch.load(f)
        with open(f"{data_dir}/router_val_data_{equation}_{boundary}_{dim}d_{in_channels}c_{arguments["n_val"]}s.pt", "rb") as f:
            val_data = torch.load(f)
    else:
        with open(f"args/grf_args.json", "r") as f:
            arguments_grf = json.load(f)

        grf = GaussianRandomField(num_samples=arguments["N"],
                                    dim=dim,
                                    alpha=arguments_grf["alpha"],
                                    beta=arguments_grf["beta"],
                                    gamma=arguments_grf["gamma"],
                                    device=device,
                                    seed=34)
        # if dim == 1:
        #     f = lambda x: np.sin(2 * np.pi * x)
        # else:
        #     f = lambda x, y: np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
        pushforward = None if boundary == "Dirichlet" else lambda x: x - torch.mean(x)
        f = grf.generate(arguments["n_train"] + arguments["n_val"] + extra, pushfoward=pushforward) if equation == "Poisson" else grf.generate(arguments["n_train"] + arguments["n_val"] + extra, pushfoward=None)
        k2 = grf.generate(arguments["n_train"] + arguments["n_val"] + extra)
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
        train_data = []
        val_data = []
        for i in range(arguments["n_train"] + arguments["n_val"] + extra):
            pde = None
            u_sol = None
            if dim == 1:
                if equation == "Poisson":
                    pde = PoissonEquation1D(a_func=a[i] if in_channels > 1 else a,
                                            f_func=f[i],
                                            boundary=boundary,
                                            x=x, 
                                            device=device)
                else:
                    pde = HelmholtzEquation1D(a_func = a[i] if in_channels > 1 else a, f_func=f[i], k2 = k2[i], boundary=boundary,x=x,device=device)
                u_sol = torch.tensor(pde.u, dtype=torch.float32, device=device)
                u_sol = u_sol - torch.mean(u_sol) if equation == "Poisson" else u_sol
            else:
                if equation == "Poisson":
                    pde = PoissonEquation2D(a_func=a[i].flatten() if in_channels > 1 else a,
                                            f_func=f[i].flatten(),
                                            boundary=boundary,
                                            x=x,
                                            y=y,
                                            device=device)
                else:
                    pde = HelmholtzEquation2D(a_func=a[i].flatten() if in_channels > 1 else a, f_func = f[i].flatten(), k2=k2[i].flatten(), boundary=boundary, x=x, y=y, device=device)
                new_shape = (arguments["N"], arguments["N"]) 
                u_sol = torch.tensor(pde.u.reshape(new_shape), dtype=torch.float32, device=device)
                u_sol = u_sol - torch.mean(u_sol) if equation == "Poisson" else u_sol
                print(f"Generated sample {i+1}/{arguments['n_train'] + arguments['n_val'] + extra}")
                print(f"len(train_data) = {len(train_data)}, len(val_data) = {len(val_data)}")
            
            if in_channels > 1:
                # input = torch.concatenate((a[i, None, :], f[i, None, :]), dim=0)
                if equation == "Poisson":
                    input = torch.concatenate((a[i, None, :], f[i, None, :]), dim=0)
                else:
                    input = torch.concatenate((a[i, None, :], k2[i, None, :], f[i, None, :]), dim=0)      
            else:
                # input = f[i, None, :]
                if equation == "Poisson":
                    input = f[i, None, :]
                else:
                    input = torch.concatenate((k2[i, None, :], f[i, None, :]), dim=0)
            residual = pde.compute_residual(u_sol.flatten())
            if torch.linalg.norm(residual) > 1:
                continue
            if len(train_data) < arguments["n_train"]:
                train_data.append((input, u_sol))
            else:
                val_data.append((input, u_sol))
            if len(train_data) == arguments["n_train"] and len(val_data) == arguments["n_val"]:
                break
        if len(train_data) < arguments["n_train"] or len(val_data) < arguments["n_val"]:
            print(f"Generated {len(train_data)} training samples and {len(val_data)} validation samples.")
            raise ValueError("Not enough data generated. Try increasing the extra variable.")
        with open(f"{data_dir}/router_train_data_{equation}_{boundary}_{dim}d_{in_channels}c_{arguments["n_train"]}s.pt", "wb") as f:
            torch.save(train_data, f)
        with open(f"{data_dir}/router_val_data_{equation}_{boundary}_{dim}d_{in_channels}c_{arguments["n_val"]}s.pt", "wb") as f:
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

    loss_fn = ApproxGreedyRouterLoss(centered=(equation == "Poisson"), normalized=False)

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