from ast import Constant
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import argparse
from ml_solver import MLSolver, DeepONet, FNOforPDE
from data_generation import GaussianRandomField, PDEDataset2, GaussianRandomFieldHierarchical
from pde import PoissonEquation1D, PoissonEquation2D, HelmholtzEquation1D, HelmholtzEquation2D
from numerical_solver import WeightedJacobiSolver, MultigridSolver, GaussSeidelSolver
from hybrid_solver import LSTMGreedyRouter, HybridSolver, ConstantRouter, HINTSRouter

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
            if model.equation.equation == "Poisson":
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
    return errors_greedy, loss_greedy, residuals, mode_1_errors, mode_5_errors, mode_10_errors, solver_decisions

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
            if model_hints.equation.equation == "Poisson":
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
    results_df_name = args.results_df_name
    results_dir = args.results_dir

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
        pushforward = None if boundary == "Dirichlet" else lambda x: x - torch.mean(x)
        f = grf.generate(n_test + extra, pushfoward=pushforward) if equation == "Poisson" and in_channels == 1 else grf.generate(n_test + extra, pushfoward=None)
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
            else:
                pde = HelmholtzEquation2D(a_func=a.reshape(-1, arguments["N"] * arguments["N"]) if in_channels > 1 else a, 
                                          f_func=f.reshape(-1, arguments["N"] * arguments["N"]), 
                                          k2=k2.reshape(-1, arguments["N"] * arguments["N"]),
                                          boundary=boundary, x=x, y=y, device=device)
            u_sol = torch.tensor(pde.u, dtype=torch.float32, device=device)
            u_sol = u_sol - torch.mean(u_sol, dim=(-2,-1), keepdim=True) if equation == "Poisson" and boundary == "Periodic" and in_channels == 1 else u_sol
        if in_channels > 1:
            if equation == "Poisson":
                input = torch.concatenate((a[:, None, :], f[:, None, :]), dim=1)
            else:
                input = torch.concatenate((a[:, None, :], k2[:, None, :], f[:, None, :]), dim=1)
        else:
            if equation == "Poisson":
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
    model.eval()
    centered = equation == "Poisson" and boundary == "Periodic" and in_channels == 1
    loss = ApproxGreedyRouterLoss(centered=centered)
    errors_greedy, loss_greedy, residuals_greedy, mode_1_errors, mode_5_errors, mode_10_errors, solver_decisions_greedy = test_model(model, test_loader, in_channels, dim, loss = loss, centered = centered, loss_t = False)

    if len(list_of_solvers) > 1:
        raise ValueError("Multiple solvers are not supported yet")
    

    constant_router=  ConstantRouter(2, 0, device = device)
    model_constant = HybridSolver(N=ml_arguments["N"], dim=dim, in_channels=in_channels, boundary=boundary, equation=pde,
                                    suite_solver=list_of_solvers+[ml_model], router=constant_router, tol=1e-7, max_iters=arguments["max_iters"], threshold=0.1).to(device)
    loss = ApproxGreedyRouterLoss(centered=(equation == "Poisson" and boundary == "Periodic"))
    errors_constant, loss_constant, residuals_constant, mode_one_constant, mode_five_constant, mode_ten_constant, solver_decisions_constant = test_model(model_constant, test_loader, in_channels, dim, loss, centered = centered, loss_t = False)

    if "jacobi" in numerical_solvers[0] or "gs" in numerical_solvers[0] or "sor" in numerical_solvers[0]:
        hints_num = 25
    else:
        hints_num = 15
    hints =  HINTSRouter(2, hints_num, device = device).to(device)
    model_hints = HybridSolver(N=ml_arguments["N"], dim=dim, in_channels=in_channels, boundary=boundary, equation=pde,
                                suite_solver=list_of_solvers+[ml_model], router=hints, tol=1e-7, max_iters=arguments["max_iters"], threshold=0.1).to(device)

    errors_hints, loss_hints, residuals_hints , mode_one_hints, mode_five_hints, mode_ten_hints, solver_decisions_hints = test_model(model_hints, test_loader, in_channels, dim, loss, equation == "Poisson")

    errors_true_greedy, best_solvers = true_greedy_model(model_hints, test_loader, in_channels, dim, loss, centered = centered, loss_t = False, max_iters = arguments["max_iters"])

    auc_greedy = np.trapezoid(errors_greedy, axis=0)
    auc_constant = np.trapezoid(errors_constant, axis=0)
    auc_true_greedy = np.trapezoid(errors_true_greedy, axis=0)
    auc_hints = np.trapezoid(errors_hints, axis=0)

    if os.path.exists(f"{results_dir}/separate_{ml_model_type}_{ml_model_name}_error_comparison.csv"):
        df_error = pd.read_csv(f"results/separate_{ml_model_type}_{ml_model_name}_error_comparison.csv")
        df_error = df_error.set_index("Methods")
    else:
        df_error = pd.DataFrame({"Methods": ["jacobi only", "HINTS-jacobi", "Learned Greedy-jacobi", "True-Greedy-jacobi","gs only", "HINTS-gs", "Learned Greedy-gs", "True-Greedy-gs", "mg only", "HINTS-mg", "Learned Greedy-mg", "True-Greedy-mg"]})

        df_error["FinalError_2d_Poisson"] = ""
        df_error["Mean_FinalError_2d_Poisson"] = ""
        df_error["Std_FinalError_2d_Poisson"] = ""
        df_error["pval_FinalError_2d_Poisson"] = ""

        df_error["AUC_Error_2d_Poisson"] = ""
        df_error["Mean_AUC_Error_2d_Poisson"] = ""
        df_error["Std_AUC_Error_2d_Poisson"] = ""
        df_error["pval_AUC_Error_2d_Poisson"] = ""

        df_error["FinalError_2d_Helmholtz"] = ""
        df_error["Mean_FinalError_2d_Helmholtz"] = ""
        df_error["Std_FinalError_2d_Helmholtz"] = ""
        df_error["pval_FinalError_2d_Helmholtz"] = ""

        df_error["AUC_Error_2d_Helmholtz"] = ""
        df_error["Mean_AUC_Error_2d_Helmholtz"] = ""
        df_error["Std_AUC_Error_2d_Helmholtz"] = ""
        df_error["pval_AUC_Error_2d_Helmholtz"] = ""

        df_error["FinalError_2d_Convdiff"] = ""
        df_error["Mean_FinalError_2d_Convdiff"] = ""
        df_error["Std_FinalError_2d_Convdiff"] = ""
        df_error["pval_FinalError_2d_Convdiff"] = ""

        df_error["AUC_Error_2d_Convdiff"] = ""
        df_error["Mean_AUC_Error_2d_Convdiff"] = ""
        df_error["Std_AUC_Error_2d_Convdiff"] = ""
        df_error["pval_AUC_Error_2d_Convdiff"] = ""

        df_error.set_index("Methods", inplace = True)
    
    df_error.loc[f"{numerical_solvers[0]} only", f"Mean_FinalError_{dim}d_{equation}"] = np.mean(errors_constant[-1]).item()
    df_error.loc[f"{numerical_solvers[0]} only", f"Std_FinalError_{dim}d_{equation}"] = np.std(errors_constant[-1]).item()
    df_error.loc[f"{numerical_solvers[0]} only", f"Mean_AUC_Error_{dim}d_{equation}"] = np.mean(auc_constant).item()
    df_error.loc[f"{numerical_solvers[0]} only", f"Std_AUC_Error_{dim}d_{equation}"] = np.std(auc_constant).item()
    df_error.loc[f"{numerical_solvers[0]} only", f"FinalError_{dim}d_{equation}"] = f"{(np.mean(errors_constant[-1])*(10**3)).item():.3f} ({(np.std(errors_constant[-1])*(10**3)).item():.3f})"
    df_error.loc[f"{numerical_solvers[0]} only", f"pval_FinalError_{dim}d_{equation}"] = ttest_rel(errors_constant[-1], errors_greedy[-1], alternative = "greater").pvalue.item()
    df_error.loc[f"{numerical_solvers[0]} only", f"AUC_Error_{dim}d_{equation}"] = f"{(np.mean(auc_constant).item()):.3f} ({(np.std(auc_constant).item()):.3f})"
    df_error.loc[f"{numerical_solvers[0]} only", f"pval_AUC_Error_{dim}d_{equation}"] = ttest_rel(auc_constant, auc_greedy, alternative = "greater").pvalue.item()

    df_error.loc[f"HINTS-{numerical_solvers[0]}", f"Mean_FinalError_{dim}d_{equation}"] = np.mean(errors_hints[-1]).item()
    df_error.loc[f"HINTS-{numerical_solvers[0]}", f"Std_FinalError_{dim}d_{equation}"] = np.std(errors_hints[-1]).item()
    df_error.loc[f"HINTS-{numerical_solvers[0]}", f"Mean_AUC_Error_{dim}d_{equation}"] = np.mean(auc_hints).item()
    df_error.loc[f"HINTS-{numerical_solvers[0]}", f"Std_AUC_Error_{dim}d_{equation}"] = np.std(auc_hints).item()
    df_error.loc[f"HINTS-{numerical_solvers[0]}", f"FinalError_{dim}d_{equation}"] = f"{(np.mean(errors_hints[-1])*(10**3)).item():.3f} ({(np.std(errors_hints[-1])*(10**3)).item():.3f})"
    df_error.loc[f"HINTS-{numerical_solvers[0]}", f"pval_FinalError_{dim}d_{equation}"] = ttest_rel(errors_hints[-1], errors_greedy[-1], alternative = "greater").pvalue.item()
    df_error.loc[f"HINTS-{numerical_solvers[0]}", f"AUC_Error_{dim}d_{equation}"] = f"{(np.mean(auc_hints).item()):.3f} ({(np.std(auc_hints).item()):.3f})"
    df_error.loc[f"HINTS-{numerical_solvers[0]}", f"pval_AUC_Error_{dim}d_{equation}"] = ttest_rel(auc_hints, auc_greedy, alternative = "greater").pvalue.item()

    df_error.loc[f"Learned Greedy-{numerical_solvers[0]}", f"Mean_FinalError_{dim}d_{equation}"] = np.mean(errors_greedy[-1]).item()
    df_error.loc[f"Learned Greedy-{numerical_solvers[0]}", f"Std_FinalError_{dim}d_{equation}"] = np.std(errors_greedy[-1]).item()
    df_error.loc[f"Learned Greedy-{numerical_solvers[0]}", f"Mean_AUC_Error_{dim}d_{equation}"] = np.mean(auc_greedy).item()
    df_error.loc[f"Learned Greedy-{numerical_solvers[0]}", f"Std_AUC_Error_{dim}d_{equation}"] = np.std(auc_greedy).item()
    df_error.loc[f"Learned Greedy-{numerical_solvers[0]}", f"FinalError_{dim}d_{equation}"] = f"{(np.mean(errors_greedy[-1])*(10**3)).item():.3f} ({(np.std(errors_greedy[-1])*(10**3)).item():.3f})"
    df_error.loc[f"Learned Greedy-{numerical_solvers[0]}", f"AUC_Error_{dim}d_{equation}"] = f"{(np.mean(auc_greedy).item()):.3f} ({(np.std(auc_greedy).item()):.3f})"
    
    df_error.loc[f"True-Greedy-{numerical_solvers[0]}", f"Mean_FinalError_{dim}d_{equation}"] = np.mean(errors_true_greedy[-1]).item()
    df_error.loc[f"True-Greedy-{numerical_solvers[0]}", f"Std_FinalError_{dim}d_{equation}"] = np.std(errors_true_greedy[-1]).item()
    df_error.loc[f"True-Greedy-{numerical_solvers[0]}", f"Mean_AUC_Error_{dim}d_{equation}"] = np.mean(auc_true_greedy).item()
    df_error.loc[f"True-Greedy-{numerical_solvers[0]}", f"Std_AUC_Error_{dim}d_{equation}"] = np.std(auc_true_greedy).item()
    df_error.loc[f"True-Greedy-{numerical_solvers[0]}", f"FinalError_{dim}d_{equation}"] = f"{(np.mean(errors_true_greedy[-1])*(10**3)).item():.3f} ({(np.std(errors_true_greedy[-1])*(10**3)).item():.3f})"
    df_error.loc[f"True-Greedy-{numerical_solvers[0]}", f"AUC_Error_{dim}d_{equation}"] = f"{(np.mean(auc_true_greedy).item()):.3f} ({(np.std(auc_true_greedy).item()):.3f})"

    df_error.to_csv(f"{results_dir}/separate_{ml_model_type}_{ml_model_name}_error_comparison.csv")

    if os.path.exists(f"{results_dir}/separate_{ml_model_type}_{ml_model_name}_residual_comparison.csv"):
        df_residual = pd.read_csv(f"{results_dir}/separate_{ml_model_type}_{ml_model_name}_residual_comparison.csv")
        df_residual = df_residual.set_index("Methods")
    else:
        df_residual = pd.DataFrame({"Methods": ["jacobi only", "HINTS-jacobi", "Learned Greedy-jacobi", "True-Greedy-jacobi","gs only", "HINTS-gs", "Learned Greedy-gs", "True-Greedy-gs", "mg only", "HINTS-mg", "Learned Greedy-mg", "True-Greedy-mg"]})

        df_residual["FinalResidual_2d_Poisson"] = ""
        df_residual["Mean_FinalResidual_2d_Poisson"] = ""
        df_residual["Std_FinalResidual_2d_Poisson"] = ""
        df_residual["pval_FinalResidual_2d_Poisson"] = ""

        df_residual["AUC_Residual_2d_Poisson"] = ""
        df_residual["Mean_AUC_Residual_2d_Poisson"] = ""
        df_residual["Std_AUC_Residual_2d_Poisson"] = ""
        df_residual["pval_AUC_Residual_2d_Poisson"] = ""

        df_residual["FinalResidual_2d_Helmholtz"] = ""
        df_residual["Mean_FinalResidual_2d_Helmholtz"] = ""
        df_residual["Std_FinalResidual_2d_Helmholtz"] = ""
        df_residual["pval_FinalResidual_2d_Helmholtz"] = ""

        df_residual["AUC_Residual_2d_Helmholtz"] = ""
        df_residual["Mean_AUC_Residual_2d_Helmholtz"] = ""
        df_residual["Std_AUC_Residual_2d_Helmholtz"] = ""
        df_residual["pval_AUC_Residual_2d_Helmholtz"] = ""


        df_residual["FinalResidual_2d_Convdiff"] = ""
        df_residual["Mean_FinalResidual_2d_Convdiff"] = ""
        df_residual["Std_FinalResidual_2d_Convdiff"] = ""
        df_residual["pval_FinalResidual_2d_Convdiff"] = ""

        df_residual["AUC_Residual_2d_Convdiff"] = ""
        df_residual["Mean_AUC_Residual_2d_Convdiff"] = ""
        df_residual["Std_AUC_Residual_2d_Convdiff"] = ""
        df_residual["pval_AUC_Residual_2d_Convdiff"] = ""
        df_residual.set_index("Methods", inplace = True)
    df_residual.loc[f"{numerical_solvers[0]} only", f"Mean_FinalResidual_{dim}d_{equation}"] = np.mean(residuals_constant[-1]).item()
    df_residual.loc[f"{numerical_solvers[0]} only", f"Std_FinalResidual_{dim}d_{equation}"] = np.std(residuals_constant[-1]).item()
    df_residual.loc[f"{numerical_solvers[0]} only", f"Mean_AUC_Residual_{dim}d_{equation}"] = np.mean(auc_constant).item()
    df_residual.loc[f"{numerical_solvers[0]} only", f"Std_AUC_Residual_{dim}d_{equation}"] = np.std(auc_constant).item()
    df_residual.loc[f"{numerical_solvers[0]} only", f"FinalResidual_{dim}d_{equation}"] = f"{(np.mean(residuals_constant[-1])*(10**3)).item():.3f} ({(np.std(residuals_constant[-1])*(10**3)).item():.3f})"
    df_residual.loc[f"{numerical_solvers[0]} only", f"pval_FinalResidual_{dim}d_{equation}"] = ttest_rel(residuals_constant[-1], residuals_greedy[-1], alternative = "greater").pvalue.item()
    df_residual.loc[f"{numerical_solvers[0]} only", f"AUC_Residual_{dim}d_{equation}"] = f"{(np.mean(auc_constant).item()):.3f} ({(np.std(auc_constant).item()):.3f})"
    df_residual.loc[f"{numerical_solvers[0]} only", f"pval_AUC_Residual_{dim}d_{equation}"] = ttest_rel(auc_constant, auc_greedy, alternative = "greater").pvalue.item()

    df_residual.loc[f"HINTS-{numerical_solvers[0]}", f"Mean_FinalResidual_{dim}d_{equation}"] = np.mean(residuals_hints[-1]).item()
    df_residual.loc[f"HINTS-{numerical_solvers[0]}", f"Std_FinalResidual_{dim}d_{equation}"] = np.std(residuals_hints[-1]).item()
    df_residual.loc[f"HINTS-{numerical_solvers[0]}", f"Mean_AUC_Residual_{dim}d_{equation}"] = np.mean(auc_hints).item()
    df_residual.loc[f"HINTS-{numerical_solvers[0]}", f"Std_AUC_Residual_{dim}d_{equation}"] = np.std(auc_hints).item()
    df_residual.loc[f"HINTS-{numerical_solvers[0]}", f"FinalResidual_{dim}d_{equation}"] = f"{(np.mean(residuals_hints[-1])*(10**3)).item():.3f} ({(np.std(residuals_hints[-1])*(10**3)).item():.3f})"
    df_residual.loc[f"HINTS-{numerical_solvers[0]}", f"pval_FinalResidual_{dim}d_{equation}"] = ttest_rel(residuals_hints[-1], residuals_greedy[-1], alternative = "greater").pvalue.item()
    df_residual.loc[f"HINTS-{numerical_solvers[0]}", f"AUC_Residual_{dim}d_{equation}"] = f"{(np.mean(auc_hints).item()):.3f} ({(np.std(auc_hints).item()):.3f})"
    df_residual.loc[f"HINTS-{numerical_solvers[0]}", f"pval_AUC_Residual_{dim}d_{equation}"] = ttest_rel(auc_hints, auc_greedy, alternative = "greater").pvalue.item()

    df_residual.loc[f"Learned Greedy-{numerical_solvers[0]}", f"Mean_FinalResidual_{dim}d_{equation}"] = np.mean(residuals_greedy[-1]).item()
    df_residual.loc[f"Learned Greedy-{numerical_solvers[0]}", f"Std_FinalResidual_{dim}d_{equation}"] = np.std(residuals_greedy[-1]).item()
    df_residual.loc[f"Learned Greedy-{numerical_solvers[0]}", f"Mean_AUC_Residual_{dim}d_{equation}"] = np.mean(auc_greedy).item()
    df_residual.loc[f"Learned Greedy-{numerical_solvers[0]}", f"Std_AUC_Residual_{dim}d_{equation}"] = np.std(auc_greedy).item()
    df_residual.loc[f"Learned Greedy-{numerical_solvers[0]}", f"FinalResidual_{dim}d_{equation}"] = f"{(np.mean(residuals_greedy[-1])*(10**3)).item():.3f} ({(np.std(residuals_greedy[-1])*(10**3)).item():.3f})"
    df_residual.loc[f"Learned Greedy-{numerical_solvers[0]}", f"AUC_Residual_{dim}d_{equation}"] = f"{(np.mean(auc_greedy).item()):.3f} ({(np.std(auc_greedy).item()):.3f})"

    df_residual.to_csv(f"{results_dir}/separate_{ml_model_type}_{ml_model_name}_residual_comparison.csv")

    # plot the solver decisions in the form of a heatmap in order of ml solver usage
    # need a discrete cmap with 2 colors from tableau10 # encode 0 as numerical solver and 1 as ml solver as a label
    # different color than these two
    titles = {"jacobi": "Jacobi", "gs": "GS", "mg": "MG"}
    cmap = colors.ListedColormap(colors.TABLEAU_COLORS.keys())
    # i need two colors f
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
    # common colorbar fo.collection[0] is giving an index error
    # common colorbar fo.collection[0] is giving an index error
    # common colorbar fo.collection[0] is giving an index error
    # common colorbar fo.collection[0] is giving an index error

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
    j = 0
    
    if numerical_solvers[0] == "gs":
        j = 1
        # two_colors = [[1, 1, 1, 1], "tab:orange"]
    elif numerical_solvers[0] == "mg":
        # two_colors = [[1, 1, 1, 1], "tab:green"]
        j = 2
    cmap = colors.ListedColormap(two_colors)
    if os.path.exists(f"{results_dir}/{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_solver_decisions_2.pkl"):
        fig, axes = pickle.load(open(f"{results_dir}/{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_solver_decisions_2.pkl", "rb"))
    else:
        fig, axes = plt.subplots(nrows = 2, ncols = 3, sharex = True, sharey = True, figsize = (15, 10))
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
    
    fig.suptitle(f"Solver Decisions for Learned Greedy and Best Solvers True Greedy")
    plt.savefig(f"{results_dir}/{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_solver_decisions_2.png")
    pickle.dump((fig, axes), open(f"{results_dir}/{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_solver_decisions_2.pkl", "wb"))
    plt.close()

    # deeponet usage plot

    if os.path.exists(f"{results_dir}/{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_deeponet_usage.pkl"):
        fig, axes = pickle.load(open(f"{results_dir}/{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_deeponet_usage.pkl", "rb"))
    else:
        fig, axes = plt.subplots(ncols = 3, sharey = True, figsize = (15, 5))

    
    
    axes[j].clear()
    axes[j].plot(np.sum(solver_decisions_greedy, axis = -1)/n_test, label = "Learned Greedy")
    axes[j].plot(np.sum(best_solvers, axis = -1)/n_test, label = "True Greedy")
    axes[j].set_xlabel("Iteration")
    axes[j].set_ylabel("DeepONet Usage (%)")
    axes[j].set_title(f"{titles[numerical_solvers[0]]}")
    axes[j].legend()
    fig.suptitle(f"DeepONet Usage for Learned Greedy and True Greedy")
    plt.savefig(f"{results_dir}/{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_deeponet_usage.png")
    pickle.dump((fig, axes), open(f"{results_dir}/{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_deeponet_usage.pkl", "wb"))
    plt.close()

    # Sample error convergence plot

    if os.path.exists(f"{results_dir}/{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_sample_error_convergence.pkl"):
        fig, axes = pickle.load(open(f"{results_dir}/{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_sample_error_convergence.pkl", "rb"))
    else:
        fig, axes = plt.subplots(ncols = 3, sharey = True, figsize = (15, 5))

    axes[j].clear()
    idx_of_interest = 43
    axes[j].plot(errors_constant[:, idx_of_interest], label = f"{titles[numerical_solvers[0]]} Only")
    axes[j].plot(errors_hints[:, idx_of_interest], label = f"HINTS-{titles[numerical_solvers[0]]}")
    axes[j].plot(errors_greedy[:, idx_of_interest], label = f"Learned Greedy-{titles[numerical_solvers[0]]}")
    # axes[j].plot(errors_true_greedy[:, idx_of_interest], label = f"True Greedy-{titles[numerical_solvers[0]]}")
    axes[j].set_xlabel("Iteration")
    axes[j].set_ylabel("Error")
    axes[j].set_title(f"{titles[numerical_solvers[0]]}")
    axes[j].set_yscale("log")
    axes[j].legend()
    fig.suptitle(f"Sample Error Convergence for Different Routing Strategies")
    plt.savefig(f"{results_dir}/{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_sample_error_convergence.png")
    pickle.dump((fig, axes), open(f"{results_dir}/{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_sample_error_convergence.pkl", "wb"))
    plt.close()