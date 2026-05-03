from ast import Constant
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import argparse
from ml_solver import MLSolver, DeepONet, FNOforPDE, DeepONetCNN
from data_generation import GaussianRandomField, PDEDataset2, GaussianRandomFieldHierarchical
from pde import PoissonEquation1D, PoissonEquation2D, HelmholtzEquation1D, HelmholtzEquation2D, ConvectionDiffusion2D
from numerical_solver import WeightedJacobiSolver, MultigridSolver, GaussSeidelSolver, SuccessiveOverRelaxationSolver, SymmetricSuccessiveOverRelaxationSolver, RichardsonSolver
from hybrid_solver import LSTMGreedyRouter, HybridSolver, ConstantRouter, HINTSRouter, LSTMGreedyRouter_SideInfo

from trainer import ApproxGreedyRouterLoss
import json
import latextable
import pickle
import pandas as pd
from texttable import Texttable
from scipy.stats import ttest_rel

parser = argparse.ArgumentParser()
parser.add_argument('--ml_model', type=str, default='deeponet', help='Model to use: deeponet or fno')
parser.add_argument('--n_test', type = int, default = 64, help = "Number of test points")
parser.add_argument("--model", type=str, default='lstm')
parser.add_argument("--extra", type=int, default=200, help="Extra data samples to generate beyond n_train + n_val")
parser.add_argument("--ckp_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
parser.add_argument("--ml_model_name", type=str, default="test", help="ml_model checkpoint name")
parser.add_argument("--model_name", type=str, default="", help="Model checkpoint name")
parser.add_argument("--data_dir", type=str, default="./data", help="Directory to save/load data")
parser.add_argument('--data_name', type=str, default='', help='Name of the dataset to use (if not provided, a new dataset will be generated)')
parser.add_argument("--grf_mode", type = str, default="fixed", help="Mode of the GRF: hierarchical or fixed")
parser.add_argument('--dim', type=int, default=1, help='Dimension of the PDE: 1 or 2')
parser.add_argument("--boundary", type=str, default="Periodic", help="Boundary condition: Dirichlet or Periodic")
parser.add_argument("--in_channels", type=int, default=1, help="Number of input channels")
parser.add_argument('--numerical_solvers', type=str, default='jacobi', help='comma-separated list of numerical solvers. Ex: jacobi_1.3,mg_2,gs')
parser.add_argument("--equation", type=str, default="Poisson", help="PDE to solve: Poisson")
parser.add_argument("--reaction_c", type=float, default=0.0, help="Reaction coefficient for Convection-Diffusion equation")
parser.add_argument("--b_vel", type=float, default=20.0, help="Advection velocity magnitude for Convection-Diffusion equation")
parser.add_argument("--results_df_name", type=str, default="", help="Name of the results dataframe")
parser.add_argument("--results_dir", type=str, default="./results", help="Path to save the results")

def test_model(model, dataloader, in_channels, dim, loss = ApproxGreedyRouterLoss(), centered = True, loss_t = False):
    model.eval()
    errors_greedy = ()
    loss_greedy = () if loss_t else None
    residuals = ()
    solver_decisions = ()
    mode_1_errors = ()
    mode_5_errors = ()
    mode_10_errors = ()
    predictions_all = ()
    output_all = ()
    with torch.no_grad():
        for batch in dataloader:
            model.reset()
            input, output = batch
            bs = input.shape[0]
            # f = input[:, 0, :].reshape(bs, -1)
            # if in_channels > 1:
            #     a = input[:, 1, :].reshape(bs, -1)
            # else:
            #     a = None
            f = input[:, -1, :].reshape(bs, -1)
            if model.equation.equation =="Poisson" or model.equation.equation == "ConvDiff":
                if in_channels > 1:
                    a = input[:, 0, :].reshape(bs, -1)
                else:
                    a = None
                k2 = None
            else:
                k2 = input[:, -2, :].reshape(bs, -1)
                if in_channels > 1:
                    a = input[:, 0, :].reshape(bs, -1)
                else:
                    a = None
            pred = model(f = f, a = a, k2 = k2, u0=None, return_dict = True, training=loss, ground_truth = output.reshape(bs, -1))
            if centered:
                predictions = pred["predictions"] - torch.mean(pred["predictions"], axis = 2, keepdim = True)
            else:
                predictions =  pred["predictions"] 
            initial_prediction = torch.zeros_like(predictions[0]).unsqueeze(0)
            predictions = torch.cat((initial_prediction, predictions), dim = 0)
            # initial_error = torch.norm()
            error = torch.norm(predictions - output.reshape(bs, -1).unsqueeze(0), dim=2).detach().cpu().numpy()
            residual = torch.norm(pred["residuals"], dim = 2).detach().cpu().numpy()
            scores = pred["routing_scores"]
            decisions = torch.argmax(scores, dim = -1).detach().cpu().numpy()
            solver_decisions += (decisions,)
            residuals += (residual, )
            errors_greedy += (error,)
            if loss_t:
                loss_greedy += (loss(pred, output.reshape(bs, -1), "none").detach().cpu().numpy(), )
            error = (predictions - output.reshape(bs, -1).unsqueeze(0)).detach().cpu().numpy()
            predictions_all += (predictions.detach().cpu().numpy(),)
            output_all += (output.detach().cpu().numpy(),)
            if dim == 1:
                mode_wise_error = np.fft.rfftn(error, axes = [-1])
                mode_1_error = mode_wise_error[:, :, 1]
                mode_1_norm = np.sqrt(mode_1_error.real**2 + mode_1_error.imag**2)
                mode_5_error = mode_wise_error[:, :, 5]
                mode_5_norm = np.sqrt(mode_5_error.real**2 + mode_5_error.imag**2)

                mode_10_error = mode_wise_error[:, :, 10]
                mode_10_norm = np.sqrt(mode_10_error.real**2 + mode_10_error.imag**2)

            else:
                N = int(np.sqrt(error.shape[2]))
                error = error.reshape(error.shape[0], error.shape[1], N, N)
                mode_wise_error = np.fft.rfftn(error, axes = [-2,-1])
                mode_1_error = mode_wise_error[:, :, 1, 1]
                mode_1_norm = np.sqrt(mode_1_error.real**2 + mode_1_error.imag**2)
                mode_5_error = mode_wise_error[:, :, 5, 5]
                mode_5_norm = np.sqrt(mode_5_error.real**2 + mode_5_error.imag**2)

                mode_10_error = mode_wise_error[:, :, 10, 10]
                mode_10_norm = np.sqrt(mode_10_error.real**2 + mode_10_error.imag**2)
            mode_1_errors += (mode_1_norm,)
            mode_5_errors += (mode_5_norm,)
            mode_10_errors += (mode_10_norm,)

    errors_greedy = np.concatenate(errors_greedy, axis = 1)
    loss_greedy = np.concatenate(loss_greedy) if loss_t else None
    residuals = np.concatenate(residuals, axis = 1)
    mode_1_errors = np.concatenate(mode_1_errors, axis = 1)
    mode_5_errors = np.concatenate(mode_5_errors, axis = 1)
    mode_10_errors = np.concatenate(mode_10_errors, axis = 1)
    solver_decisions = np.concatenate(solver_decisions, axis = 1)
    output_all = np.concatenate(output_all, axis = 0)
    predictions_all = np.concatenate(predictions_all, axis = 1)
    return errors_greedy, loss_greedy, residuals, mode_1_errors, mode_5_errors, mode_10_errors, solver_decisions, predictions_all, output_all

def true_greedy_model(model_hints, test_loader, in_channels, dim, loss = ApproxGreedyRouterLoss(), centered = True, loss_t = False, max_iters = 100):
    errors_true_greedy = ()
    loss_true_greedy = ()
    best_solvers = ()
    with torch.no_grad():
        for batch in test_loader:
            model_hints.reset()
            input, output = batch
            bs = input.shape[0]
            f = input[:, -1, :].reshape(bs, -1)
            if model_hints.equation.equation =="Poisson" or model_hints.equation.equation == "ConvDiff":
                if in_channels > 1:
                    a = input[:, 0, :].reshape(bs, -1)
                else:
                    a = None
                k2 = None
            else:
                k2 = input[:, -2, :].reshape(bs, -1)
                if in_channels > 1:
                    a = input[:, 0, :].reshape(bs, -1)
                else:
                    a = None
            
            pred = model_hints(f = f, a = a, k2=k2, u0=None, return_dict = True, training=True, teacher_forcing = 1.0, ground_truth = output.reshape(bs, -1))
            if centered:
                expert_predictions = pred["complete_expert_predictions"] - torch.mean(pred["complete_expert_predictions"], dim=-1, keepdim=True)
            else:
                expert_predictions = pred["complete_expert_predictions"]
            # initial_prediction = torch.zeros_like(expert_predictions[0]).unsqueeze(0)
            # expert_predictions = torch.cat((initial_prediction, expert_predictions), dim = 0)
            errors_expert = torch.norm(expert_predictions - output.reshape(bs, -1).unsqueeze(0), dim=-1)
            best_solver = torch.argmin(errors_expert, dim=1)
            scores = torch.zeros(max_iters, input.shape[0], 2, device = device)
            scores = torch.nn.functional.one_hot(best_solver, num_classes=2).to(scores.dtype)
            pred["routing_scores"] = scores 
            if loss_t:
                loss_true_greedy += (loss(pred, output.reshape(bs, -1), "none").detach().cpu().numpy(), )
            initial_prediction = torch.zeros_like(expert_predictions[0]).unsqueeze(0)
            expert_predictions = torch.cat((initial_prediction, expert_predictions), dim = 0)
            errors_expert = torch.norm(expert_predictions - output.reshape(bs, -1).unsqueeze(0), dim=-1)
            best_error = torch.min(errors_expert, dim=1).values.detach().cpu().numpy()
            errors_true_greedy += (best_error,)
            best_solvers += (best_solver.detach().cpu().numpy(),)
    errors_true_greedy = np.concatenate(errors_true_greedy, axis=1)
    if loss_t:
        loss_true_greedy = np.concatenate(loss_true_greedy)
    else:
        loss_true_greedy = None
    best_solvers = np.concatenate(best_solvers, axis=1)
    return errors_true_greedy, best_solvers

def clear_axes_with_colorbar(ax):
    if len(ax.images) > 0:
        im = ax.images[0]
        cbar = getattr(im, "colorbar", None)
        if cbar is not None:
            cbar.remove()
    ax.clear()


if __name__ == "__main__":
    print("Parsing arguments...")
    args, unknown = parser.parse_known_args()
    ml_model_type = args.ml_model
    n_test = args.n_test
    model_type = args.model
    extra = args.extra
    ckp_dir = args.ckp_dir
    ml_model_name = args.ml_model_name
    model_name = args.model_name
    data_dir = args.data_dir
    data_name = args.data_name
    grf_mode = args.grf_mode
    numerical_solvers = args.numerical_solvers.split(",")
    num_solvers = len(numerical_solvers) + 1
    dim = args.dim
    boundary = args.boundary
    in_channels = args.in_channels
    equation = args.equation
    reaction_c = args.reaction_c
    b_vel = args.b_vel
    results_df_name = args.results_df_name
    results_dir = args.results_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if boundary not in ["Periodic", "Dirichlet"]:
        raise ValueError("Boundary condition must be either 'Dirichlet' or 'Periodic'")
    if equation not in ["Poisson", "Helmholtz", "ConvDiff"]:
        raise ValueError("Currently only Poisson/Helmholtz/ConvDiff equation is supported")
    if ml_model_type not in ["deeponet", "fno", "deeponetcnn"]:
        raise ValueError("Model must be either 'deeponet' or 'fno' or 'deeponetcnn'")
    if dim not in [1, 2]:
        raise ValueError("Dimension must be either 1 or 2")
    if in_channels not in [1, 2]:
        raise ValueError("in_channels must be either 1 or 2")
    if model_type not in ["lstm", "lstm_side_info"]:
        raise ValueError("Model must be LSTM")
    
    ml_ckp_path = ckp_dir + f"/{ml_model_type}_{ml_model_name}_{grf_mode}_{equation}_{boundary}_{dim}d_{in_channels}c_best.pth"
    ml_args_path = ckp_dir + f"/{ml_model_type}_{ml_model_name}_{grf_mode}_{equation}_{boundary}_{dim}d_{in_channels}c_args.json"

    ckp_path = ckp_dir + f"/{model_type}router_{model_name}_{grf_mode}_{equation}_{boundary}_{dim}d_{in_channels}c_{args.numerical_solvers}_best.pth"
    save_path = ckp_dir + f"/{model_type}router_{model_name}_{grf_mode}_{equation}_{boundary}_{dim}d_{in_channels}c_{args.numerical_solvers}"
    args_path = ckp_dir + f"/{model_type}router_{model_name}_{grf_mode}_{equation}_{boundary}_{dim}d_{in_channels}c_{args.numerical_solvers}args.json"
    
    if os.path.exists(args_path):
        print(f"Loading training arguments from {args_path}...")
        with open(args_path, "r") as f:
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
    if os.path.exists(f"{data_dir}/{data_name}router_test_data_{grf_mode}_{equation}_{boundary}_{dim}d_{in_channels}c_{n_test}s.pt"):
        print(f"Loading data from {data_dir}...")
        with open(f"{data_dir}/{data_name}router_test_data_{grf_mode}_{equation}_{boundary}_{dim}d_{in_channels}c_{n_test}s.pt", "rb") as f:
            test_data = torch.load(f)
    else:
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
                                                    device=device, seed=72)
        else:
            with open(f"args/grf_args.json", "r") as f:
                arguments_grf = json.load(f)
            
            grf = GaussianRandomField(num_samples=arguments["N"],
                                    dim=dim,
                                    alpha=arguments_grf["alpha"],
                                    beta=arguments_grf["beta"],
                                    gamma=arguments_grf["gamma"],
                                    device=device, seed=72)
        if dim == 1:
            pushforward = lambda x: x - torch.mean(x, dim=-1, keepdim=True)
        else:
            pushforward = lambda x: x - torch.mean(x, dim=(-2, -1), keepdim=True)
        needs_mean_zero = (equation == "Poisson") or (equation == "ConvDiff" and reaction_c == 0.0)
        f = grf.generate(n_test + extra, pushfoward=pushforward) if needs_mean_zero and boundary == "Periodic" and in_channels == 1 else grf.generate(n_test + extra, pushfoward=None)
        k2 = grf.generate(n_test + extra)
        if in_channels > 1:
            a = grf.generate(n_test + extra)
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
            u_sol = u_sol - torch.mean(u_sol, dim = -1, keepdim=True) if equation == "Poisson" and boundary == "Periodic" and in_channels == 1 else u_sol
        else:
            if equation == "Poisson":
                pde = PoissonEquation2D(a_func=a.reshape(-1, arguments["N"] * arguments["N"]) if in_channels > 1 else a, 
                                        f_func=f.reshape(-1, arguments["N"] * arguments["N"]),
                                        boundary=boundary, 
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
                                          f_func=f.reshape(-1, arguments["N"] * arguments["N"]), 
                                          k2=k2.reshape(-1, arguments["N"] * arguments["N"]),
                                          boundary=boundary, x=x, y=y, device=device)
            u_sol = torch.tensor(pde.u, dtype=torch.float32, device=device)
            u_sol = u_sol - torch.mean(u_sol, dim=(-2,-1), keepdim=True) if needs_mean_zero and boundary == "Periodic" and in_channels == 1 else u_sol
        if in_channels > 1:
            if equation == "Poisson" or equation == "ConvDiff":
                input = torch.concatenate((a[:, None, :], f[:, None, :]), dim=1)
            else:
                input = torch.concatenate((a[:, None, :], k2[:, None, :], f[:, None, :]), dim=1)
        else:
            if equation == "Poisson" or equation == "ConvDiff":
                input = f[:, None, :]
            else:
                input = torch.concatenate((k2[:, None, :], f[:, None, :]), dim=1)
        test_data = [input[:n_test], u_sol[:n_test]]
        with open(f"{data_dir}/{data_name}router_test_data_{grf_mode}_{equation}_{boundary}_{dim}d_{in_channels}c_{n_test}s.pt", "wb") as f:
            torch.save(test_data, f)
    print("Data creation/loading completed.")
    print(f"Test data size: {test_data[0].shape[0]}")
    print(f"Size of each input: {test_data[0][0].shape}, Size of each solution: {test_data[1][0].shape}")
    # Change this later 
    test_dataset = PDEDataset2(test_data)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=arguments["batch_size"], shuffle=False)
    print(f"Test dataset size: {len(test_dataset)}")


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
    elif ml_model_type == "deeponetcnn":
        ml_model = DeepONetCNN(N=ml_arguments["N"], dim=dim, in_channels=new_in_channels, device=device, boundary=boundary,
                        branch_dim=ml_arguments["branch_dim"],
                        hidden_branch_channels=ml_arguments["hidden_branch_channels"],
                        kernel_size=ml_arguments["kernel_size"],
                        stride=ml_arguments["stride"],
                        hidden_trunk=ml_arguments["hidden_trunk"],
                        hidden_branch=ml_arguments["hidden_branch"]).to(device)
    else:
        raise ValueError("Invalid ML Model Type")
    
    
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
    elif model_type == "lstm_side_info":
        if dim == 1:
            router = LSTMGreedyRouter_SideInfo(None, ml_arguments["N"]*(new_in_channels + 1), arguments["hidden_dim"], arguments["num_layers"], num_solvers, arguments["dropout"]).to(device)
        else:
            router = LSTMGreedyRouter_SideInfo(None, ml_arguments["N"]*ml_arguments["N"]*(new_in_channels + 1), arguments["hidden_dim"], arguments["num_layers"], num_solvers, arguments["dropout"]).to(device)

    
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
                                    b_vec=(b_vel, b_vel),
                                    boundary=boundary,
                                    x=x, y=y, device=device, 
                                    reaction=reaction_c,
                                    solve = False)
    else:
        if dim == 1:
            pde = HelmholtzEquation1D(a_func= lambda x: 1, f_func=lambda x: 1, k2=lambda x: 1, boundary=boundary, x=x, device=device, solve = False)
        else:
            pde = HelmholtzEquation2D(a_func=lambda x, y: 1,f_func=lambda x,y: 1, k2=lambda x,y: 1, boundary=boundary,x=x,y=y, device=device, solve = False)


    list_of_solvers = []
    for solver in numerical_solvers:
        split = solver.split("_")
        print(split[0])
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
        elif split[0] == "sor":
            if len(split) > 1:
                omega = float(split[1])
            else:
                omega = 1.0
            list_of_solvers.append(SuccessiveOverRelaxationSolver(pde, device, omega))
        elif split[0] == "ssor":
            if len(split) > 1:
                omega = float(split[1])
            else:
                omega = 1.0
            list_of_solvers.append(SymmetricSuccessiveOverRelaxationSolver(pde, device, omega))
        elif split[0] == "rich":
            if len(split) > 1:
                omega = float(split[1])
            else:
                omega = 1.0
            list_of_solvers.append(RichardsonSolver(pde, device, omega))

        else:
            raise ValueError("Invalid Numerical Solver")
    print(f"List of solvers: {list_of_solvers}")

    model = HybridSolver(N=arguments["N"], dim=dim, in_channels=in_channels, boundary=boundary, equation=pde,
                                    suite_solver=list_of_solvers+[ml_model], router=router, tol=1e-7, max_iters=arguments["max_iters"], threshold=0.1)
    model.eval()
    needs_mean_zero = (equation == "Poisson") or (equation == "ConvDiff" and reaction_c == 0.0)
    centered = needs_mean_zero and boundary == "Periodic" and in_channels == 1
    loss = ApproxGreedyRouterLoss(centered=centered)
    errors_greedy, loss_greedy, residuals_greedy, mode_1_errors, mode_5_errors, mode_10_errors, solver_decisions_greedy, pred_greedy, _ = test_model(model, test_loader, in_channels, dim, loss = loss, centered = centered, loss_t = False)

    if len(list_of_solvers) > 1:
        raise ValueError("Multiple solvers are not supported yet")
    

    constant_router=  ConstantRouter(2, 0, device = device)
    model_constant = HybridSolver(N=ml_arguments["N"], dim=dim, in_channels=in_channels, boundary=boundary, equation=pde,
                                    suite_solver=list_of_solvers+[ml_model], router=constant_router, tol=1e-7, max_iters=arguments["max_iters"], threshold=0.1).to(device)
    loss = ApproxGreedyRouterLoss(centered=(equation == "Poisson" and boundary == "Periodic"))
    errors_constant, loss_constant, residuals_constant, mode_one_constant, mode_five_constant, mode_ten_constant, solver_decisions_constant, pred_constant, _ = test_model(model_constant, test_loader, in_channels, dim, loss, centered = centered, loss_t = False)

    if "jacobi" in numerical_solvers[0] or "gs" in numerical_solvers[0] or "sor" in numerical_solvers[0] or "ssor" in numerical_solvers[0]:
        hints_num = 25
    else:
        hints_num = 15
    hints =  HINTSRouter(2, hints_num, device = device).to(device)
    model_hints = HybridSolver(N=ml_arguments["N"], dim=dim, in_channels=in_channels, boundary=boundary, equation=pde,
                                suite_solver=list_of_solvers+[ml_model], router=hints, tol=1e-7, max_iters=arguments["max_iters"], threshold=0.1).to(device)

    errors_hints, loss_hints, residuals_hints , mode_one_hints, mode_five_hints, mode_ten_hints, solver_decisions_hints, pred_hints, output_hints = test_model(model_hints, test_loader, in_channels, dim, loss, equation == "Poisson")

    errors_true_greedy, best_solvers = true_greedy_model(model_hints, test_loader, in_channels, dim, loss, centered = centered, loss_t = False, max_iters = arguments["max_iters"])

    auc_greedy = np.trapezoid(errors_greedy, axis=0)
    auc_constant = np.trapezoid(errors_constant, axis=0)
    auc_true_greedy = np.trapezoid(errors_true_greedy, axis=0)
    auc_hints = np.trapezoid(errors_hints, axis=0)

    titles = {"jacobi": "Jacobi", "gs": "GS", "mg": "MG", "sor": "SOR", "ssor": "SymGS", "rich": "Richardson", "jacobi_0.67": "Jacobi (0.67)", "sor_1.5": "SOR (1.5)"}
    method_list = []
    for method in ["jacobi", "gs", "mg", "ssor", "jacobi_0.67", "sor_1.5"]:
        for t in ["only ", "HINTS-", "Learned Greedy-", "True-Greedy-"]:
            method_list.append(f"{t}{titles[method]}")


    if os.path.exists(f"{results_dir}/{results_df_name}_separate_{ml_model_type}_{ml_model_name}_error_comparison.csv"):
        df_error = pd.read_csv(f"{results_dir}/{results_df_name}_separate_{ml_model_type}_{ml_model_name}_error_comparison.csv")
        df_error = df_error.set_index("Methods")
    else:
        df_error = pd.DataFrame({"Methods": method_list})

        for e in ["Poisson", "Helmholtz", "ConvDiff"]:
            df_error[f"FinalError_{dim}d_{e}"] = ""
            df_error[f"Mean_FinalError_{dim}d_{e}"] = ""
            df_error[f"Std_FinalError_{dim}d_{e}"] = ""
            df_error[f"pval_FinalError_{dim}d_{e}"] = ""
            df_error[f"AUC_Error_{dim}d_{e}"] = ""
            df_error[f"Mean_AUC_Error_{dim}d_{e}"] = ""
            df_error[f"Std_AUC_Error_{dim}d_{e}"] = ""
            df_error[f"pval_AUC_Error_{dim}d_{e}"] = ""

        df_error.set_index("Methods", inplace = True)
    errors_dict = {"only ": errors_constant, "HINTS-": errors_hints, "Learned Greedy-": errors_greedy, "True-Greedy-": errors_true_greedy}
    for key, value in errors_dict.items():
        auc_value = np.trapezoid(value, axis=0)
        df_error.loc[f"{key}{titles[numerical_solvers[0]]}", f"Mean_FinalError_{dim}d_{equation}"] = np.mean(value[-1]).item()
        df_error.loc[f"{key}{titles[numerical_solvers[0]]}", f"Std_FinalError_{dim}d_{equation}"] = np.std(value[-1]).item()
        df_error.loc[f"{key}{titles[numerical_solvers[0]]}", f"Mean_AUC_Error_{dim}d_{equation}"] = np.mean(auc_value).item()
        df_error.loc[f"{key}{titles[numerical_solvers[0]]}", f"Std_AUC_Error_{dim}d_{equation}"] = np.std(auc_value).item()
        df_error.loc[f"{key}{titles[numerical_solvers[0]]}", f"FinalError_{dim}d_{equation}"] = f"{(np.mean(value[-1])*(10**3)).item():.3f} ({(np.std(value[-1])*(10**3)).item():.3f})"
        df_error.loc[f"{key}{titles[numerical_solvers[0]]}", f"pval_FinalError_{dim}d_{equation}"] = ttest_rel(value[-1], errors_greedy[-1], alternative = "greater").pvalue.item()
        df_error.loc[f"{key}{titles[numerical_solvers[0]]}", f"AUC_Error_{dim}d_{equation}"] = f"{(np.mean(auc_value).item()):.3f} ({(np.std(auc_value).item()):.3f})"
        df_error.loc[f"{key}{titles[numerical_solvers[0]]}", f"pval_AUC_Error_{dim}d_{equation}"] = ttest_rel(auc_value, auc_greedy, alternative = "greater").pvalue.item()

   
    df_error.to_csv(f"{results_dir}/{results_df_name}_separate_{ml_model_type}_{ml_model_name}_error_comparison.csv")
    new_method_list = []
    
    for method in ["jacobi", "gs", "mg", "ssor", "jacobi_0.67", "sor_1.5"]:
        for t in ["only ", "HINTS-", "Learned Greedy-"]:
            new_method_list.append(f"{t}{titles[method]}")
    if os.path.exists(f"{results_dir}/{results_df_name}_separate_{ml_model_type}_{ml_model_name}_residual_comparison.csv"):
        df_residual = pd.read_csv(f"{results_dir}/{results_df_name}_separate_{ml_model_type}_{ml_model_name}_residual_comparison.csv")
        df_residual = df_residual.set_index("Methods")
    else:
        df_residual = pd.DataFrame({"Methods": new_method_list})
        for e in ["Poisson", "Helmholtz", "ConvDiff"]:
            df_residual[f"FinalResidual_{dim}d_{e}"] = ""
            df_residual[f"Mean_FinalResidual_{dim}d_{e}"] = ""
            df_residual[f"Std_FinalResidual_{dim}d_{e}"] = ""
            df_residual[f"pval_FinalResidual_{dim}d_{e}"] = ""
            df_residual[f"AUC_Residual_{dim}d_{e}"] = ""
            df_residual[f"Mean_AUC_Residual_{dim}d_{e}"] = ""
            df_residual[f"Std_AUC_Residual_{dim}d_{e}"] = ""
            df_residual[f"pval_AUC_Residual_{dim}d_{e}"] = ""
        df_residual.set_index("Methods", inplace = True)
    auc_residual_constant = np.trapezoid(residuals_constant, axis=0)
    auc_residual_hints = np.trapezoid(residuals_hints, axis=0)
    auc_residual_greedy = np.trapezoid(residuals_greedy, axis=0)
    # auc_residual_true_greedy = np.trapezoid(residuals_true_greedy, axis=0)
    residuals_dict = {"only ": residuals_constant, "HINTS-": residuals_hints, "Learned Greedy-": residuals_greedy}
    for key, value in residuals_dict.items():
        auc_value = np.trapezoid(value, axis=0)
        df_residual.loc[f"{key}{titles[numerical_solvers[0]]}", f"Mean_FinalResidual_{dim}d_{equation}"] = np.mean(value[-1]).item()
        df_residual.loc[f"{key}{titles[numerical_solvers[0]]}", f"Std_FinalResidual_{dim}d_{equation}"] = np.std(value[-1]).item()
        df_residual.loc[f"{key}{titles[numerical_solvers[0]]}", f"Mean_AUC_Residual_{dim}d_{equation}"] = np.mean(auc_value).item()
        df_residual.loc[f"{key}{titles[numerical_solvers[0]]}", f"Std_AUC_Residual_{dim}d_{equation}"] = np.std(auc_value).item()
        df_residual.loc[f"{key}{titles[numerical_solvers[0]]}", f"FinalResidual_{dim}d_{equation}"] = f"{(np.mean(value[-1])*(10**3)).item():.3f} ({(np.std(value[-1])*(10**3)).item():.3f})"
        df_residual.loc[f"{key}{titles[numerical_solvers[0]]}", f"pval_FinalResidual_{dim}d_{equation}"] = ttest_rel(value[-1], residuals_greedy[-1], alternative = "greater").pvalue.item()
        df_residual.loc[f"{key}{titles[numerical_solvers[0]]}", f"AUC_Residual_{dim}d_{equation}"] = f"{(np.mean(auc_value).item()):.3f} ({(np.std(auc_value).item()):.3f})"
        df_residual.loc[f"{key}{titles[numerical_solvers[0]]}", f"pval_AUC_Residual_{dim}d_{equation}"] = ttest_rel(auc_value, auc_greedy, alternative = "greater").pvalue.item()
   
    df_residual.to_csv(f"{results_dir}/{results_df_name}_separate_{ml_model_type}_{ml_model_name}_residual_comparison.csv")

    auc_mode_one = np.trapezoid(mode_1_errors, axis=0)
    auc_mode_five = np.trapezoid(mode_5_errors, axis=0)
    auc_mode_ten = np.trapezoid(mode_10_errors, axis=0)

    auc_mode_one_constant = np.trapezoid(mode_one_constant, axis=0)
    auc_mode_five_constant = np.trapezoid(mode_five_constant, axis=0)
    auc_mode_ten_constant = np.trapezoid(mode_ten_constant, axis=0)

    auc_mode_one_hints = np.trapezoid(mode_one_hints, axis=0)
    auc_mode_five_hints = np.trapezoid(mode_five_hints, axis=0)
    auc_mode_ten_hints = np.trapezoid(mode_ten_hints, axis=0)


    if os.path.exists(f"{results_dir}/{results_df_name}_separate_{ml_model_type}_{ml_model_name}_mode_comparison.csv"):
        df_mode = pd.read_csv(f"{results_dir}/{results_df_name}_separate_{ml_model_type}_{ml_model_name}_mode_comparison.csv")
        df_mode = df_mode.set_index("Methods")
    else:
        df_mode = pd.DataFrame({"Methods": new_method_list})
        for e in ["Poisson", "Helmholtz", "ConvDiff"]:
            for m in ["1", "5", "10"]:
                df_mode[f"Final_Mode_{m}_Error_{dim}d_{e}"] = ""
                df_mode[f"Mean_Final_Mode_{m}_{dim}d_{e}"] = ""
                df_mode[f"Std_Final_Mode_{m}_{dim}d_{e}"] = ""
                df_mode[f"pval_Final_Mode_{m}_{dim}d_{e}"] = ""
                df_mode[f"AUC_Mode_{m}_{dim}d_{e}"] = ""
                df_mode[f"Mean_AUC_Mode_{m}_{dim}d_{e}"] = ""
                df_mode[f"Std_AUC_Mode_{m}_{dim}d_{e}"] = ""
                df_mode[f"pval_AUC_Mode_{m}_{dim}d_{e}"] = ""
        df_mode.set_index("Methods", inplace = True)
    
    mode_one_dict = {"only ": mode_one_constant, "HINTS-": mode_one_hints, "Learned Greedy-": mode_1_errors}
    mode_five_dict = {"only ": mode_five_constant, "HINTS-": mode_five_hints, "Learned Greedy-": mode_5_errors}
    mode_ten_dict = {"only ": mode_ten_constant, "HINTS-": mode_ten_hints, "Learned Greedy-": mode_10_errors}
    mode_dict = {"1": mode_one_dict, "5": mode_five_dict, "10": mode_ten_dict}
    for m, mode_num_dict in mode_dict.items():
        auc_greedy_mode = np.trapezoid(mode_num_dict["Learned Greedy-"], axis=0)
        for key, value in mode_num_dict.items():
            auc_value = np.trapezoid(value, axis=0)
            df_mode.loc[f"{key}{titles[numerical_solvers[0]]}", f"Mean_Final_Mode_{m}_{dim}d_{equation}"] = np.nanmean(value[-1]).item()
            df_mode.loc[f"{key}{titles[numerical_solvers[0]]}", f"Std_Final_Mode_{m}_{dim}d_{equation}"] = np.nanstd(value[-1]).item()
            df_mode.loc[f"{key}{titles[numerical_solvers[0]]}", f"pval_Final_Mode_{m}_{dim}d_{equation}"] = ttest_rel(value[-1], mode_num_dict["Learned Greedy-"][-1], alternative = "greater").pvalue.item()
            df_mode.loc[f"{key}{titles[numerical_solvers[0]]}", f"Mean_AUC_Mode_{m}_{dim}d_{equation}"] = np.nanmean(auc_value).item()
            df_mode.loc[f"{key}{titles[numerical_solvers[0]]}", f"Std_AUC_Mode_{m}_{dim}d_{equation}"] = np.nanstd(auc_value).item()
            df_mode.loc[f"{key}{titles[numerical_solvers[0]]}", f"pval_AUC_Mode_{m}_{dim}d_{equation}"] = ttest_rel(auc_value, auc_greedy_mode, alternative = "greater").pvalue.item()
            df_mode.loc[f"{key}{titles[numerical_solvers[0]]}", f"Final_Mode_{m}_Error_{dim}d_{equation}"] = f"{(np.nanmean(value[-1])*(10**3)).item():.3f} ({(np.nanstd(value[-1])*(10**3)).item():.3f})"
            df_mode.loc[f"{key}{titles[numerical_solvers[0]]}", f"AUC_Mode_{m}_{dim}d_{equation}"] = f"{(np.nanmean(auc_value).item()):.3f} ({(np.nanstd(auc_value).item()):.3f})"
    
    df_mode.to_csv(f"{results_dir}/{results_df_name}_separate_{ml_model_type}_{ml_model_name}_mode_comparison.csv")
    

    
    cmap = colors.ListedColormap(colors.TABLEAU_COLORS.keys())
    two_colors = cmap(np.linspace(0,1,2))
    two_colors = ["tab:blue", "tab:orange"] 
    
    cmap = colors.ListedColormap(two_colors)
    
    fig, axes = plt.subplots(nrows=2, sharex=True, figsize = (10, 10))
    # make the plots take more space
    axes[0].set_aspect("auto")
    axes[1].set_aspect("auto")
    axes[0].imshow(solver_decisions_greedy.transpose(), cmap = cmap)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Sample")
    axes[0].set_title("Solver Decisions for Learned Greedy")

    axes[1].imshow(best_solvers.transpose(), cmap = cmap)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Sample")
    axes[1].set_title("Best Solvers True Greedy")

    cbar = fig.colorbar(axes[0].imshow(solver_decisions_greedy.transpose(), cmap = cmap), ax = axes, location = "right")
    # shorten distance bwteen cbar and labels
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.yaxis.set_label_coords(1.15, 0.5)
    cbar.set_label("Solver Decision")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([numerical_solvers[0], "ML Solver"])
    fig.suptitle(f"Solver Decisions for Learned Greedy and Best Solvers True Greedy for {numerical_solvers[0]}")
    # fig.tight_layout()
    plt.savefig(f"{results_dir}/separate_{ml_model_type}_{ml_model_name}_{numerical_solvers[0]}_{equation}_{boundary}_{dim}d_{in_channels}c_solver_decisions.png")
    plt.close()

    # if the pickle file exists, load it otherwise create a new figure
    j = -1
    appendix = False
    if numerical_solvers[0] == "gs":
        j = 1
        # two_colors = [[1, 1, 1, 1], "tab:orange"]
    elif numerical_solvers[0] == "mg" or numerical_solvers[0] == "ssor":
        # two_colors = [[1, 1, 1, 1], "tab:green"]
        j = 2
    elif numerical_solvers[0] == "jacobi":
        j = 0
    else:
        appendix = True
        if numerical_solvers[0] == "jacobi_0.67":
            j = 0
        else:
            j = 1
    cmap = colors.ListedColormap(two_colors)

    # SOLVER DECISIONS PLOT
    if not appendix:
        if os.path.exists(f"{results_dir}/{results_df_name}_{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_solver_decisions_2.pkl"):
            fig, axes = pickle.load(open(f"{results_dir}/{results_df_name}_{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_solver_decisions_2.pkl", "rb"))
        else:
            fig, axes = plt.subplots(nrows = 2, ncols = 3, sharex = True, sharey = True, figsize = (15, 10))
            cbar = fig.colorbar(axes[0, j].imshow(solver_decisions_greedy.transpose(), cmap = cmap, aspect = 7), ax = axes, location = "right")
            cbar.ax.yaxis.set_label_position("right")
            cbar.set_ticks([0.25, 0.75])
            cbar.set_ticklabels(["Numerical Solver", "ML Solver"])
    else:
        if os.path.exists(f"{results_dir}/{results_df_name}_{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_solver_decisions_2_appendix.pkl"):
            fig, axes = pickle.load(open(f"{results_dir}/{results_df_name}_{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_solver_decisions_2_appendix.pkl", "rb"))
        else:
            fig, axes = plt.subplots(nrows = 2, ncols = 2, sharex = True, sharey = True, figsize = (10, 10))
            cbar = fig.colorbar(axes[0, j].imshow(solver_decisions_greedy.transpose(), cmap = cmap, aspect = 7), ax = axes, location = "right")
            cbar.ax.yaxis.set_label_position("right")
            cbar.set_ticks([0.25, 0.75])
            cbar.set_ticklabels(["Numerical Solver", "ML Solver"])

    axes[0, j].clear()
    axes[0, j].imshow(solver_decisions_greedy.transpose(), cmap = cmap, aspect = 7)
    axes[0, j].set_xlabel("Iteration")
    axes[0, j].set_ylabel("Sample")
    axes[0, j].set_title(f"Learned Greedy {titles[numerical_solvers[0]]}")

    axes[1, j].clear()
    axes[1, j].imshow(best_solvers.transpose(), cmap = cmap, aspect = 7)
    axes[1, j].set_xlabel("Iteration")
    axes[1, j].set_ylabel("Sample")
    axes[1, j].set_title(f"True Greedy {titles[numerical_solvers[0]]}")
    
    fig.suptitle(f"Solver Decisions for Learned Greedy and True Greedy for {equation}")
    if appendix:
        plt.savefig(f"{results_dir}/{results_df_name}_{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_solver_decisions_2_appendix.png")
        pickle.dump((fig, axes), open(f"{results_dir}/{results_df_name}_{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_solver_decisions_2_appendix.pkl", "wb"))
    else:   
        plt.savefig(f"{results_dir}/{results_df_name}_{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_solver_decisions_2.png")
        pickle.dump((fig, axes), open(f"{results_dir}/{results_df_name}_{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_solver_decisions_2.pkl", "wb"))
    plt.close()
    
    # Deeponet usage plots for both equations
    offset = -0.02 if appendix else 0
    if not appendix:
        if os.path.exists(f"{results_dir}/{results_df_name}_{ml_model_type}_deeponet_usage.pkl"):
            fig, axes = pickle.load(open(f"{results_dir}/{results_df_name}_{ml_model_type}_deeponet_usage.pkl", "rb"))
        else:
            fig, axes = plt.subplots(ncols = 3, nrows = 2, sharey = True, figsize = (15, 10))
            row_titles = ['Poisson', 'ConvDiff']
            y = [0.29, 0.7]
            
            for k, title in enumerate(row_titles):
                fig.text(0.07 + offset, y[k], title, rotation=90, va='center', ha='center', fontsize=18)
    else:
        if os.path.exists(f"{results_dir}/{results_df_name}_{ml_model_type}_deeponet_usage_appendix.pkl"):
            fig, axes = pickle.load(open(f"{results_dir}/{results_df_name}_{ml_model_type}_deeponet_usage_appendix.pkl", "rb"))
        else:
            fig, axes = plt.subplots(ncols = 2, nrows = 2, sharey = True, figsize = (10, 10))
            row_titles = ['Poisson', 'ConvDiff']
            y = [0.29, 0.7]
            for k, title in enumerate(row_titles):
                fig.text(0.07 + offset, y[k], title, rotation=90, va='center', ha='center', fontsize=18)
    if equation == "Poisson":
        i = 1
    else:
        i = 0
    
    axes[i, j].clear()
    axes[i, j].plot(np.sum(solver_decisions_greedy, axis = -1)/n_test, label = "Learned Greedy")
    axes[i, j].plot(np.sum(best_solvers, axis = -1)/n_test, label = "True Greedy")
    axes[i, j].set_xlabel("Iteration")
    axes[i, j].set_ylabel("DeepONet Usage (%)")
    axes[i, j].set_title(f"{titles[numerical_solvers[0]]}", fontsize = 18, pad = 20)
    axes[i, j].legend()
    fig.suptitle(f"DeepONet Usage: Learned Greedy v/s True Greedy", fontsize = 24)
    if appendix:
        plt.savefig(f"{results_dir}/{results_df_name}_{ml_model_type}_deeponet_usage_appendix.png")
        pickle.dump((fig, axes), open(f"{results_dir}/{results_df_name}_{ml_model_type}_deeponet_usage_appendix.pkl", "wb"))
    else:
        plt.savefig(f"{results_dir}/{results_df_name}_{ml_model_type}_deeponet_usage.png")
        pickle.dump((fig, axes), open(f"{results_dir}/{results_df_name}_{ml_model_type}_deeponet_usage.pkl", "wb"))
    plt.close()

    # new_colors = {"jacobi": "tab:blue", "gs": "tab:orange", "mg": "tab:green", "ssor": "tab:red", "jacobi_0.67": "tab:purple", "sor_1.5": "tab:brown"}
    # if not appendix:
    #     if os.path.exists(f"{results_dir}/{results_df_name}_{ml_model_type}_deeponet_usage_learned_greedy.pkl"):
    #         fig, axes = pickle.load(open(f"{results_dir}/{results_df_name}_{ml_model_type}_deeponet_usage_learned_greedy.pkl", "rb"))
    #     else:
    #         fig, axes = plt.subplots(ncols = 2, sharey = True, figsize = (10, 5))
    
    #     axes[i].plot(np.sum(solver_decisions_greedy, axis = -1)[:100]/n_test, label = f"Greedy {titles[numerical_solvers[0]]}", color = new_colors[numerical_solvers[0]])
    #     axes[i].set_xlabel("Iteration")
    #     axes[i].set_ylabel("DeepONet Usage (%)")
    #     axes[i].set_title(f"{equation}", fontsize = 18)
    #     if axes[i].get_legend() is not None:
    #         axes[i].get_legend().remove()
    #     axes[i].legend()
    #     fig.suptitle(f"DeepONet Usage", fontsize = 24)
    #     if not appendix:
    #         plt.savefig(f"{results_dir}/{results_df_name}_{ml_model_type}_deeponet_usage_learned_greedy.png")
    #         pickle.dump((fig, axes), open(f"{results_dir}/{results_df_name}_{ml_model_type}_deeponet_usage_learned_greedy.pkl", "wb"))
    #         plt.close()
    #     else:
    #         plt.savefig(f"{results_dir}/{results_df_name}_{ml_model_type}_deeponet_usage_learned_greedy_appendix.png")
    #         pickle.dump((fig, axes), open(f"{results_dir}/{results_df_name}_{ml_model_type}_deeponet_usage_learned_greedy_appendix.pkl", "wb"))
    #         plt.close()

    # Sample error convergence plot


    idx_of_interest = 29 #95
    if not appendix:
        if os.path.exists(f"{results_dir}/{results_df_name}_{ml_model_type}_sample_error_convergence.pkl"):
            fig, axes = pickle.load(open(f"{results_dir}/{results_df_name}_{ml_model_type}_sample_error_convergence.pkl", "rb"))
        else:
            fig, axes = plt.subplots(ncols = 3, nrows = 2, sharey = True, figsize = (15, 10))
            row_titles = ['Poisson', 'ConvDiff']
            y = [0.29, 0.7]
            for k, title in enumerate(row_titles):
                fig.text(0.07 + offset, y[k], title, rotation=90, va='center', ha='center', fontsize=18)
    else:
        if os.path.exists(f"{results_dir}/{results_df_name}_{ml_model_type}_sample_error_convergence_appendix.pkl"):
            fig, axes = pickle.load(open(f"{results_dir}/{results_df_name}_{ml_model_type}_sample_error_convergence_appendix.pkl", "rb"))
        else:
            fig, axes = plt.subplots(ncols = 2, nrows = 2, sharey = True, figsize = (10, 10))
            row_titles = ['Poisson', 'ConvDiff']
            y = [0.29, 0.7]
            for k, title in enumerate(row_titles):
                fig.text(0.07 + offset, y[k], title, rotation=90, va='center', ha='center', fontsize=18)
    
    axes[i, j].clear()
    axes[i, j].plot(errors_constant[:, idx_of_interest], label = f"{titles[numerical_solvers[0]]} Only")
    axes[i, j].plot(errors_hints[:, idx_of_interest], label = f"HINTS-{titles[numerical_solvers[0]]}")
    axes[i, j].plot(errors_greedy[:, idx_of_interest], label = f"Learned Greedy-{titles[numerical_solvers[0]]}")
    # axes[i, j].plot(errors_true_greedy[:, idx_of_interest], label = f"True Greedy-{titles[numerical_solvers[0]]}")
    axes[i, j].set_xlabel("Iteration")
    axes[i, j].set_ylabel("Error")
    axes[i, j].set_title(f"{titles[numerical_solvers[0]]}", fontsize = 18, pad = 20)
    axes[i, j].set_yscale("log")
    axes[i, j].legend()
    fig.suptitle(f"Sample Error Convergence for Different Routing Strategies", fontsize = 24)
    if appendix:
        plt.savefig(f"{results_dir}/{results_df_name}_{ml_model_type}_sample_error_convergence_appendix.png")
        pickle.dump((fig, axes), open(f"{results_dir}/{results_df_name}_{ml_model_type}_sample_error_convergence_appendix.pkl", "wb"))
    else:
        plt.savefig(f"{results_dir}/{results_df_name}_{ml_model_type}_sample_error_convergence.png")
        pickle.dump((fig, axes), open(f"{results_dir}/{results_df_name}_{ml_model_type}_sample_error_convergence.pkl", "wb"))
    
    # Sample visaulization plots

    predictions = {"only ": pred_constant[-1, idx_of_interest].reshape(arguments["N"], arguments["N"]), "HINTS-": pred_hints[-1, idx_of_interest].reshape(arguments["N"], arguments["N"]), "Learned Greedy-": pred_greedy[-1, idx_of_interest].reshape(arguments["N"], arguments["N"])}
    output_question = output_hints[idx_of_interest].reshape(arguments["N"], arguments["N"])
    if not appendix:
        if os.path.exists(f"{results_dir}/{results_df_name}_{ml_model_type}_{equation}_{boundary}_{dim}d_{in_channels}c_sample_visualization.pkl"):
            fig, axes = pickle.load(open(f"{results_dir}/{results_df_name}_{ml_model_type}_{equation}_{boundary}_{dim}d_{in_channels}c_sample_visualization.pkl", "rb"))
        else:
            fig, axes = plt.subplots(ncols = 4, nrows = 3, sharey = True, sharex = True, figsize = (20, 15)) # ncols, nrows
            row_titles = ["Greedy","HINTS",  "Solver Only"]
            y = [0.23, 0.5, 0.77]
            for k, title in enumerate(row_titles):
                fig.text(0.09 + offset, y[k], title, rotation=90, va='center', ha='center', fontsize=18)
            axes[0, -1].set_axis_off()
            axes[2, -1].set_axis_off()
            norm = colors.Normalize(vmin = np.min(output_question), vmax = np.max(output_question))
            trueee = axes[1, -1].imshow(output_question, norm = norm,  extent = [0, 1, 0, 1])
            axes[1, -1].set_title("Truth", fontsize = 18)
            cbar = fig.colorbar(trueee, ax=axes[:, :], fraction=0.25, pad=0.02)

    else:
        if os.path.exists(f"{results_dir}/{results_df_name}_{ml_model_type}_{equation}_{boundary}_{dim}d_{in_channels}c_sample_visualization_appendix.pkl"):
            fig, axes = pickle.load(open(f"{results_dir}/{results_df_name}_{ml_model_type}_{equation}_{boundary}_{dim}d_{in_channels}c_sample_visualization_appendix.pkl", "rb"))
        else:
            fig, axes = plt.subplots(ncols = 3, nrows = 3, sharey = True, sharex = True, figsize = (15, 15))
            row_titles = ["Greedy","HINTS",  "Solver Only"]
            y = [0.23, 0.5, 0.77]
            for k, title in enumerate(row_titles):
                fig.text(0.09 + offset, y[k], title, rotation=90, va='center', ha='center', fontsize=18)
            axes[0, -1].set_axis_off()
            axes[2, -1].set_axis_off()
            # TRUE
            norm = colors.Normalize(vmin = np.min(output_question), vmax = np.max(output_question))
            trueee = axes[1, -1].imshow(output_question, norm = norm,  extent = [0, 1, 0, 1])
            axes[1, -1].set_title("Truth", fontsize = 18)
            cbar = fig.colorbar(trueee, ax=axes[:, :], fraction=0.25, pad=0.02)
            

    

    norm = colors.Normalize(vmin = np.min(output_question), vmax = np.max(output_question))
    # solver only
    axes[0, j].clear()
    axes[0, j].set_title(titles[numerical_solvers[0]], fontsize = 18, pad = 26)
    axes[0, j].imshow(predictions["only "], norm = norm,  extent = [0, 1, 0, 1])

    axes[1, j].clear()
    axes[1, j].imshow(predictions["HINTS-"], norm = norm,  extent = [0, 1, 0, 1])

    axes[2, j].clear()
    axes[2, j].imshow(predictions["Learned Greedy-"], norm = norm,  extent = [0, 1, 0, 1])
    fig.suptitle(f"Sample Predictions for {equation}", fontsize = 24, x= 0.4)
    if not appendix:
        plt.savefig(f"{results_dir}/{results_df_name}_{ml_model_type}_{equation}_{boundary}_{dim}d_{in_channels}c_sample_visualization.png")
        pickle.dump((fig, axes), open(f"{results_dir}/{results_df_name}_{ml_model_type}_{equation}_{boundary}_{dim}d_{in_channels}c_sample_visualization.pkl", "wb"))
        plt.close()
    else:
        plt.savefig(f"{results_dir}/{results_df_name}_{ml_model_type}_{equation}_{boundary}_{dim}d_{in_channels}c_sample_visualization_appendix.png")
        pickle.dump((fig, axes), open(f"{results_dir}/{results_df_name}_{ml_model_type}_{equation}_{boundary}_{dim}d_{in_channels}c_sample_visualization_appendix.pkl", "wb"))
        plt.close()

    # Error Visualization Plots
    if not appendix:
        if os.path.exists(f"{results_dir}/{results_df_name}_{ml_model_type}_{equation}_{boundary}_{dim}d_{in_channels}c_error_visualization.pkl"):
            fig, axes = pickle.load(open(f"{results_dir}/{results_df_name}_{ml_model_type}_{equation}_{boundary}_{dim}d_{in_channels}c_error_visualization.pkl", "rb"))
        else:
            fig, axes = plt.subplots(ncols = 3, nrows = 3, sharey = True, sharex = True, figsize = (15, 15))
            row_titles = ["Greedy","HINTS", "Solver Only"]
            y = [0.23, 0.5, 0.77]
            for k, title in enumerate(row_titles):
                fig.text(0.09 + offset, y[k], title, rotation=90, va='center', ha='center', fontsize=18)
            
    else:
        if os.path.exists(f"{results_dir}/{results_df_name}_{ml_model_type}_{equation}_{boundary}_{dim}d_{in_channels}c_error_visualization_appendix.pkl"):
            fig, axes = pickle.load(open(f"{results_dir}/{results_df_name}_{ml_model_type}_{equation}_{boundary}_{dim}d_{in_channels}c_error_visualization_appendix.pkl", "rb"))
        else:
            fig, axes = plt.subplots(ncols = 2, nrows = 3, sharey = True, sharex = True, figsize = (10, 15))
            row_titles = ["Greedy","HINTS", "Solver Only"]
            y = [0.23, 0.5, 0.77]
            for k, title in enumerate(row_titles):
                fig.text(0.09 + offset, y[k], title, rotation=90, va='center', ha='center', fontsize=18)
    

       
    clear_axes_with_colorbar(axes[0, j])
    only = axes[0, j].imshow(output_question - predictions["only "], extent = [0, 1, 0, 1])
    cbar = axes[0, j].figure.colorbar(only, ax = axes[0, j])
    axes[0, j].set_title(titles[numerical_solvers[0]], fontsize = 18, pad = 26)


    clear_axes_with_colorbar(axes[1, j])
    hints = axes[1, j].imshow(output_question - predictions["HINTS-"], extent = [0, 1, 0, 1])
    cbar2= axes[1, j].figure.colorbar(hints, ax = axes[1, j])

    clear_axes_with_colorbar(axes[2, j])
    greedy = axes[2, j].imshow(output_question - predictions["Learned Greedy-"], extent = [0, 1, 0, 1])
    cbar3 = axes[2, j].figure.colorbar(greedy, ax = axes[2, j])

    fig.suptitle(f"Error of Sample Predictions for {equation}", fontsize = 24)

    if not appendix:
        plt.savefig(f"{results_dir}/{results_df_name}_{ml_model_type}_{equation}_{boundary}_{dim}d_{in_channels}c_error_visualization.png")
        pickle.dump((fig, axes), open(f"{results_dir}/{results_df_name}_{ml_model_type}_{equation}_{boundary}_{dim}d_{in_channels}c_error_visualization.pkl", "wb"))
        plt.close()
    else:
        plt.savefig(f"{results_dir}/{results_df_name}_{ml_model_type}_{equation}_{boundary}_{dim}d_{in_channels}c_error_visualization_appendix.png")
        pickle.dump((fig, axes), open(f"{results_dir}/{results_df_name}_{ml_model_type}_{equation}_{boundary}_{dim}d_{in_channels}c_error_visualization_appendix.pkl", "wb"))
        plt.close()