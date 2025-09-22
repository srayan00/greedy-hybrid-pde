import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from ml_solver import MLSolver, DeepONet, FNOforPDE
from data_generation import GaussianRandomField, PDEDataset
from pde_pytorch import PoissonEquation1D, PoissonEquation2D, HelmholtzEquation1D, HelmholtzEquation2D
from numerical_solver_pytorch import WeightedJacobiSolver, MultigridSolver, GaussSeidelSolver
from hybrid_solver import Router, ConstantRouter, HINTSRouter, LSTMGreedyRouter, HybridSolver


from trainer import Trainer, EarlyStopping, ApproxGreedyRouterLoss, ScheduledSampler
import json
import latextable
import pickle
import pandas as pd
from texttable import Texttable

parser = argparse.ArgumentParser()
parser.add_argument('--ml_model', type=str, default='deeponet', help='Model to use: deeponet or fno')
parser.add_argument('--n_test', type = int, default = 64, help = "Number of test points")
parser.add_argument("--model", type=str, default='lstm')
parser.add_argument("--extra", type=int, default=200, help="Extra data samples to generate beyond n_train + n_val")
parser.add_argument("--ckp_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
parser.add_argument("--ml_model_name", type=str, default="test", help="ml_model checkpoint name")
parser.add_argument("--model_name", type=str, default="", help="Model checkpoint name")
parser.add_argument("--data_dir", type=str, default="./data", help="Directory to save/load data")


def test_model(model, dataloader, in_channels, dim, loss = ApproxGreedyRouterLoss(), centered = True, loss_t = False):
    model.eval()
    errors_greedy = ()
    loss_greedy = () if loss_t else None
    residuals = ()
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
            error = torch.norm(predictions - output.reshape(bs, -1).unsqueeze(0), dim=2).detach().cpu().numpy()
            residual = torch.norm(pred["residuals"], dim = 2).detach().cpu().numpy()
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
    return errors_greedy, loss_greedy, residuals, mode_1_errors, mode_5_errors, mode_10_errors


def run(model_type, ml_model_type, n_test, dim, boundary, equation, ckp_dir, model_name, ml_model_name, data_dir, in_channels, device, extra = 1000):
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    ml_ckp_path = ckp_dir + f"/{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_best.pth"
    ml_args_path = ckp_dir + f"/{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_args.json"

    args_path = ckp_dir + f"/{model_type}router_{model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_jacobiargs.json"

    if os.path.exists(args_path):
        print(f"Loading training arguments from {args_path}...")
        with open(args_path, "r") as f:
            arguments = json.load(f)
    
    if os.path.exists(ml_args_path):
        with open(ml_args_path, "r") as f:
            ml_arguments = json.load(f)
    else:
        raise ValueError("ML Args Path Not found")
    # Creating/Loading Data
    print("Creating Data")

    test_or_train = "test"

    if os.path.exists(f"{data_dir}/router_{test_or_train}_data_{equation}_{boundary}_{dim}d_{in_channels}c_{n_test}s.pt"):
        print(f"Loading data from {data_dir}...")
        with open(f"{data_dir}/router_{test_or_train}_data_{equation}_{boundary}_{dim}d_{in_channels}c_{n_test}s.pt", "rb") as f:
            test_data = torch.load(f)
    else:
        with open(f"args/grf_args.json", "r") as f:
            arguments_grf = json.load(f)

        grf = GaussianRandomField(num_samples=arguments["N"],
                                    dim=dim,
                                    alpha=arguments_grf["alpha"],
                                    beta=arguments_grf["beta"],
                                    gamma=arguments_grf["gamma"],
                                    device=device,
                                    seed = 2134)
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
        test_data = []
        for i in range(n_test + extra):
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
                    pde = HelmholtzEquation2D(a_func=a[i].flatten() if in_channels > 1 else a, f_func = f[i].flatten(), k2=k2[i].flatten(),boundary=boundary, x=x, y=y, device=device)
                new_shape = (arguments["N"], arguments["N"]) 
                u_sol = torch.tensor(pde.u.reshape(new_shape), dtype=torch.float32, device=device)
                u_sol = u_sol - torch.mean(u_sol) if equation == "Poisson" else u_sol
            
            if in_channels > 1:
                if equation == "Poisson":
                    input = torch.concatenate((a[i, None, :], f[i, None, :]), dim=0)
                else:
                    input = torch.concatenate((a[i, None, :], k2[i, None, :], f[i, None, :]), dim=0)                        
            else:
                if equation == "Poisson":
                    input = f[i, None, :]
                else:
                    input = torch.concatenate((k2[i, None, :], f[i, None, :]), dim=0)
            residual = pde.compute_residual(u_sol.flatten())
            if torch.linalg.norm(residual) > 1:
                continue
            test_data.append((input, u_sol))
            if len(test_data) == n_test:
                break
        if len(test_data) < n_test:
            print(f"Generated {len(test_data)} test samples")
            raise ValueError("Not enough data generated. Try increasing the extra variable.")
        with open(f"{data_dir}/router_test_data_{equation}_{boundary}_{dim}d_{in_channels}c_{arguments["n_train"]}s.pt", "wb") as f:
            torch.save(test_data, f)
    print("Data creation/loading completed.")
    print(f"Test data size: {len(test_data)}")
    print(f"Size of each input: {test_data[0][0].shape}, Size of each solution: {test_data[0][1].shape}")
    # Change this later 
    test_dataset = PDEDataset(test_data)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=arguments["batch_size"], shuffle=True)
    # test_dataset = PDEDataset(test_data[:2])
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False)
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
    

    ckp_path = ckp_dir + f"/{model_type}router_{model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_jacobi_best.pth"
    save_path = ckp_dir + f"/{model_type}router_{model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_jacobi"
    args_path = ckp_dir + f"/{model_type}router_{model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_jacobiargs.json"

    if os.path.exists(args_path):
        print(f"Loading training arguments from {args_path}...")
        with open(args_path, "r") as f:
            arguments = json.load(f)
    else:
        raise ValueError("Router Args Path Not found")
    if model_type == "lstm":
        print(f"HERE")
        if dim == 1:
            router = LSTMGreedyRouter(None, ml_arguments["N"]*(new_in_channels + 1), arguments["hidden_dim"], arguments["num_layers"], 2, arguments["dropout"]).to(device)
        else:
            router = LSTMGreedyRouter(None, ml_arguments["N"]*ml_arguments["N"]*(new_in_channels + 1), arguments["hidden_dim"], arguments["num_layers"], 2, arguments["dropout"]).to(device)
        # router = LSTMGreedyRouter(None, ml_arguments["N"]*(in_channels + 1), arguments["hidden_dim"], arguments["num_layers"], num_solvers, arguments["dropout"]).to(device)
    ckp = None
    if os.path.exists(ckp_path):
        print(f"Loading model checkpoint from {ckp_path}...")
        ckp = torch.load(ckp_path, map_location=device)
    
    if ckp:
        print(f"Loading router state dict...")
        router.load_state_dict(ckp["model"])
        print(f"Loaded router state dict")
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

    model = HybridSolver(N=arguments["N"], dim=dim, in_channels=in_channels, boundary=boundary, equation=pde,
                                    suite_solver=[WeightedJacobiSolver(pde, device)]+[ml_model], router=router, tol=1e-7, max_iters=arguments["max_iters"], threshold=0.1).to(device)
    
    
    model.eval()
    loss = ApproxGreedyRouterLoss(centered=(equation == "Poisson"))
    errors_greedy, loss_greedy, residuals_greedy, mode_one_greedy, mode_five_greedy, mode_ten_greedy = test_model(model, test_loader, in_channels, dim,loss, equation == "Poisson")

    constant_jacobi =  ConstantRouter(2, 0, device = device)
    model_constant_jacobi = HybridSolver(N=arguments["N"], dim=dim, in_channels=in_channels, boundary=boundary, equation=pde,
                                  suite_solver=[WeightedJacobiSolver(pde, device)]+[ml_model], router=constant_jacobi, tol=1e-7, max_iters=arguments["max_iters"], threshold=0.1).to(device)

    errors_constant_jacobi, loss_constant_jacobi, residuals_constant_jacobi, mode_one_constant_jacobi, mode_five_constant_jacobi, mode_ten_constant_jacobi = test_model(model_constant_jacobi, test_loader, in_channels, dim, loss, equation == "Poisson")
    

    gs = GaussSeidelSolver(pde, device)
    constant_gs =  ConstantRouter(2, 0, device = device)
    model_constant_gs = HybridSolver(N=arguments["N"], dim=dim, in_channels=in_channels, boundary=boundary, equation=pde,
                                  suite_solver=[gs]+[ml_model], router=constant_gs, tol=1e-7, max_iters=arguments["max_iters"], threshold=0.1).to(device)
    
    errors_constant_gs, loss_constant_gs, residuals_constant_gs , mode_one_constant_gs, mode_five_constant_gs, mode_ten_constant_gs = test_model(model_constant_gs, test_loader, in_channels, dim, loss, equation == "Poisson")

    hints =  HINTSRouter(2, 25, device = device).to(device)
    model_hints = HybridSolver(N=arguments["N"], dim=dim, in_channels=in_channels, boundary=boundary, equation=pde,
                                  suite_solver=[WeightedJacobiSolver(pde, device)]+[ml_model], router=hints, tol=1e-7, max_iters=arguments["max_iters"], threshold=0.1).to(device)
    
    errors_hints, loss_hints, residuals_hints , mode_one_hints, mode_five_hints, mode_ten_hints = test_model(model_hints, test_loader, in_channels, dim, loss, equation == "Poisson")

    hints =  HINTSRouter(2, 25, device = device).to(device)
    model_hints_gs = HybridSolver(N=arguments["N"], dim=dim, in_channels=in_channels, boundary=boundary, equation=pde,
                                  suite_solver=[gs]+[ml_model], router=hints, tol=1e-7, max_iters=arguments["max_iters"], threshold=0.1).to(device)
    
    errors_hints_gs, loss_hints_gs, residuals_hints_gs ,mode_one_hints_gs, mode_five_hints_gs, mode_ten_hints_gs = test_model(model_hints_gs, test_loader, in_channels, dim, loss, equation == "Poisson")

    ckp_path = ckp_dir + f"/{model_type}router_{model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_gs_best.pth"
    args_path = ckp_dir + f"/{model_type}router_{model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_gsargs.json"
    if os.path.exists(args_path):
        print(f"Loading training arguments from {args_path}...")
        with open(args_path, "r") as f:
            arguments = json.load(f)
    else:
        raise ValueError("GS Router Args Path Not found")
    
    if model_type == "lstm":
        print(f"HERE")
        if dim == 1:
            router_gs = LSTMGreedyRouter(None, ml_arguments["N"]*(new_in_channels + 1), arguments["hidden_dim"], arguments["num_layers"], 2, arguments["dropout"]).to(device)
        else:
            router_gs = LSTMGreedyRouter(None, ml_arguments["N"]*ml_arguments["N"]*(new_in_channels + 1), arguments["hidden_dim"], arguments["num_layers"], 2, arguments["dropout"]).to(device)

    ckp = None

    if os.path.exists(ckp_path):
        print(f"Loading GS model checkpoint from {ckp_path}...")
        ckp = torch.load(ckp_path, map_location=device)
    
    if ckp:
        print(f"Loading GS router state dict...")
        router_gs.load_state_dict(ckp["model"])
        print(f"Loaded router state dict")

    model_greedy_gs = HybridSolver(N=arguments["N"], dim=dim, in_channels=in_channels, boundary=boundary, equation=pde,
                                    suite_solver=[gs]+[ml_model], router=router_gs, tol=1e-7, max_iters=arguments["max_iters"], threshold=0.1).to(device)
    
    model_greedy_gs.eval()
    errors_greedy_gs, loss_greedy_gs, residuals_greedy_gs, mode_one_greedy_gs, mode_five_greedy_gs, mode_ten_greedy_gs = test_model(model_greedy_gs, test_loader, in_channels, dim, loss, equation == "Poisson")

    temp_model_name = model_name
    ckp_path = ckp_dir + f"/{model_type}router_{temp_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_gs,jacobi_best.pth"
    args_path = ckp_dir + f"/{model_type}router_{temp_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_gs,jacobiargs.json"
    if os.path.exists(args_path):
        print(f"Loading training arguments from {args_path}...")
        with open(args_path, "r") as f:
            arguments = json.load(f)
    else:
        raise ValueError("GS,Jacobi Router Args Path Not found")

    if model_type == "lstm":
        print(f"HERE")
        if dim == 1:
            router_gs_jac = LSTMGreedyRouter(None, ml_arguments["N"]*(new_in_channels + 1), arguments["hidden_dim"], arguments["num_layers"], 3, arguments["dropout"]).to(device)
        else:
            router_gs_jac = LSTMGreedyRouter(None, ml_arguments["N"]*ml_arguments["N"]*(new_in_channels + 1), arguments["hidden_dim"], arguments["num_layers"], 3, arguments["dropout"]).to(device)
    ckp = None

    if os.path.exists(ckp_path):
        print(f"Loading GS,Jacobi model checkpoint from {ckp_path}...")
        ckp = torch.load(ckp_path, map_location=device)
    
    if ckp:
        print(f"Loading GS,Jaocbi router state dict...")
        router_gs_jac.load_state_dict(ckp["model"])
        print(f"Loaded router state dict")

    model_greedy_gs_jac = HybridSolver(N=arguments["N"], dim=dim, in_channels=in_channels, boundary=boundary, equation=pde,
                                    suite_solver=[gs]+[WeightedJacobiSolver(pde, device)] +[ml_model], router=router_gs_jac, tol=1e-7, max_iters=arguments["max_iters"], threshold=0.1).to(device)
    
    model_greedy_gs_jac.eval()
    errors_greedy_gs_jac, loss_greedy_gs_jac, residuals_greedy_gs_jac, mode_one_greedy_gs_jac, mode_five_greedy_gs_jac, mode_ten_greedy_gs_jac = test_model(model_greedy_gs_jac, test_loader, in_channels, dim, loss, equation == "Poisson")

    ckp_path = ckp_dir + f"/{model_type}router_{model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_mg_best.pth"
    args_path = ckp_dir + f"/{model_type}router_{model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_mgargs.json"
    if os.path.exists(args_path):
        print(f"Loading training arguments from {args_path}...")
        with open(args_path, "r") as f:
            arguments = json.load(f)
    else:
        raise ValueError("MG Router Args Path Not found")

    if model_type == "lstm":
        print(f"HERE")
        if dim == 1:
            router_mg = LSTMGreedyRouter(None, ml_arguments["N"]*(new_in_channels + 1), arguments["hidden_dim"], arguments["num_layers"], 2, arguments["dropout"]).to(device)
        else:
            router_mg = LSTMGreedyRouter(None, ml_arguments["N"]*ml_arguments["N"]*(new_in_channels + 1), arguments["hidden_dim"], arguments["num_layers"], 2, arguments["dropout"]).to(device)
    ckp = None

    if os.path.exists(ckp_path):
        print(f"Loading MG model checkpoint from {ckp_path}...")
        ckp = torch.load(ckp_path, map_location=device)
    
    if ckp:
        print(f"Loading MG router state dict...")
        router_mg.load_state_dict(ckp["model"])
        print(f"Loaded router state dict")

    mg = MultigridSolver(pde, 2, device)
    model_greedy_mg = HybridSolver(N=arguments["N"], dim=dim, in_channels=in_channels, boundary=boundary, equation=pde,
                                    suite_solver=[mg]+[ml_model], router=router_mg, tol=1e-7, max_iters=arguments["max_iters"], threshold=0.1).to(device)
    
    model_greedy_mg.eval()
    errors_greedy_mg, loss_greedy_mg, residuals_greedy_mg, mode_one_greedy_mg, mode_five_greedy_mg, mode_ten_greedy_mg = test_model(model_greedy_mg, test_loader, in_channels, dim, loss, equation == "Poisson")

    constant_mg =  ConstantRouter(2, 0, device = device)
    model_constant_mg = HybridSolver(N=arguments["N"], dim=dim, in_channels=in_channels, boundary=boundary, equation=pde,
                                  suite_solver=[mg]+[ml_model], router=constant_mg, tol=1e-7, max_iters=arguments["max_iters"], threshold=0.1).to(device)
    
    errors_constant_mg, loss_constant_mg, residuals_constant_mg, mode_one_constant_mg, mode_five_constant_mg, mode_ten_constant_mg = test_model(model_constant_mg, test_loader, in_channels, dim, loss, equation == "Poisson")

    hints =  HINTSRouter(2, 15, device = device).to(device)
    model_hints_mg = HybridSolver(N=arguments["N"], dim=dim, in_channels=in_channels, boundary=boundary, equation=pde,
                                  suite_solver=[mg]+[ml_model], router=hints, tol=1e-7, max_iters=arguments["max_iters"], threshold=0.1).to(device)
    
    errors_hints_mg, loss_hints_mg, residuals_hints_mg, mode_one_hints_mg, mode_five_hints_mg, mode_ten_hints_mg = test_model(model_hints_mg, test_loader, in_channels, dim, loss, equation == "Poisson")


    errors_true_greedy = ()
    loss_true_greedy = ()
    with torch.no_grad():
        for batch in test_loader:
            model_hints.reset()
            input, output = batch
            bs = input.shape[0]
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
            
            pred = model_hints(f = f, a = a, k2=k2, u0=None, return_dict = True, training=True, teacher_forcing = 1.0, ground_truth = output.reshape(bs, -1))
            if equation == "Poisson":
                expert_predictions = pred["complete_expert_predictions"] - torch.mean(pred["complete_expert_predictions"], dim=-1, keepdim=True)
            else:
                expert_predictions = pred["complete_expert_predictions"]
            errors_expert = torch.norm(expert_predictions - output.reshape(bs, -1).unsqueeze(0), dim=-1)
            best_solver = torch.argmin(errors_expert, dim=1)
            scores = torch.zeros(arguments["max_iters"], input.shape[0], 2, device = device)
            scores = torch.nn.functional.one_hot(best_solver, num_classes=2).to(scores.dtype)
            pred["routing_scores"] = scores 
            best_error = torch.min(errors_expert, dim=1).values.detach().cpu().numpy()
            errors_true_greedy += (best_error,)
            loss_true_greedy += (loss(pred, output.reshape(bs, -1), "none").detach().cpu().numpy(), )
    errors_true_greedy = np.concatenate(errors_true_greedy, axis=1)
    loss_true_greedy = np.concatenate(loss_true_greedy)


    plt.figure(figsize=(10, 5))
    plt.plot(np.mean(errors_greedy, axis=1), label='Greedy-Jacobi')
    plt.plot(np.mean(errors_constant_jacobi, axis=1), label='Jacobi Only')
    plt.plot(np.mean(errors_constant_gs, axis=1), label='GS Only')
    plt.plot(np.mean(errors_hints, axis=1), label='HINTS-Jacobi')
    plt.plot(np.mean(errors_hints_gs, axis=1), label='HINTS-GS')
    plt.plot(np.mean(errors_greedy_gs, axis=1), label='Greedy-GS')
    plt.plot(np.mean(errors_greedy_gs_jac, axis=1), label='Greedy-Jacobi,GS')
    # plt.plot(np.mean(errors_true_greedy, axis=1), label='True Greedy Router')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.yscale("log")
    plt.legend()
    plt.title('Error Comparison of Different Routing Strategies')
    plt.savefig(f"plots/{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_error_comparison_{model_type}router_{model_name}.png")

    plt.figure(figsize=(10, 5))
    plt.plot(np.mean(residuals_greedy, axis=1), label='Greedy-Jacobi')
    plt.plot(np.mean(residuals_constant_jacobi, axis=1), label='Jacobi Only')
    plt.plot(np.mean(residuals_constant_gs, axis=1), label='GS Only')
    plt.plot(np.mean(residuals_hints, axis=1), label='HINTS-Jacobi')
    plt.plot(np.mean(residuals_hints_gs, axis=1), label='HINTS-GS')
    plt.plot(np.mean(residuals_greedy_gs, axis=1), label='Greedy-GS')
    plt.plot(np.mean(residuals_greedy_gs_jac, axis=1), label='Greedy-Jacobi,GS')
    # plt.plot(np.mean(errors_true_greedy, axis=1), label='True Greedy Router')
    plt.xlabel('Iteration')
    plt.ylabel('Residual')
    plt.yscale("log")
    plt.legend()
    plt.title('Residual Comparison of Different Routing Strategies')
    plt.savefig(f"plots/{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_residual_comparison_{model_type}router_{model_name}.png")

    plt.figure(figsize=(10, 5))
    plt.plot(np.mean(errors_greedy_mg, axis=1), label='Greedy-MG')
    plt.plot(np.mean(errors_constant_mg, axis=1), label='MG Only')
    plt.plot(np.mean(errors_hints_mg, axis=1), label='HINTS-MG')
    # plt.plot(np.mean(errors_true_greedy, axis=1), label='True Greedy Router')
    plt.xlabel('V Cycles')
    plt.ylabel('Error')
    plt.yscale("log")
    plt.legend()
    plt.title('Error Comparison of Different Routing Strategies')
    plt.savefig(f"plots/MG_{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_error_comparison_{model_type}router_{model_name}.png")


    fig, axes = plt.subplots(ncols=3, sharey="row", figsize = (15, 5))
    plt.yscale("log")
    # axes[0].plot(np.mean(errors_constant_jacobi, axis=1), label='Solver Only')
    # axes[0].plot(np.mean(errors_hints, axis=1), label='HINTS')
    # axes[0].plot(np.mean(errors_greedy, axis=1), label='Greedy')
    axes[0].plot(errors_constant_jacobi[:, 20], label='Solver Only')
    axes[0].plot(errors_hints[:, 20], label='HINTS')
    axes[0].plot(errors_greedy[:, 20], label='Greedy')
    axes[0].set_title("Jacobi")
    axes[0].set_xlabel("Iterations")

    # axes[1].plot(np.mean(errors_constant_gs, axis=1), label='Solver Only')
    # axes[1].plot(np.mean(errors_hints_gs, axis=1), label='HINTS')
    # axes[1].plot(np.mean(errors_greedy_gs, axis=1), label='Greedy')
    # axes[1].plot(np.mean(errors_greedy_gs_jac, axis=1), label='Greedy-GS,Jacobi')
    axes[1].plot(errors_constant_gs[:, 20], label='Solver Only')
    axes[1].plot(errors_hints_gs[:, 20], label='HINTS')
    axes[1].plot(errors_greedy_gs[:, 20], label='Greedy')
    axes[1].plot(errors_greedy_gs_jac[:, 20], label='Greedy-GS,Jacobi')
    axes[1].set_title("GS")
    axes[1].set_xlabel("Iterations")

    # axes[2].plot(np.mean(errors_constant_mg, axis=1), label='Solver Only')
    # axes[2].plot(np.mean(errors_hints_mg, axis=1), label='HINTS')
    # axes[2].plot(np.mean(errors_greedy_mg, axis=1), label='Greedy')
    axes[2].plot(errors_constant_mg[:, 20], label='Solver Only')
    axes[2].plot(errors_hints_mg[:, 20], label='HINTS')
    axes[2].plot(errors_greedy_mg[:, 20], label='Greedy')
    axes[2].set_title("MG")
    axes[2].set_xlabel("V cycles")

    line, label = axes[1].get_legend_handles_labels()
    fig.legend(line, label, loc="center right")

    plt.suptitle('Error Comparison of Different Routing Strategies')

    plt.savefig(f"plots/separate_{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_error_comparison_{model_type}router_{model_name}.png")
    with open(f"results/{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_error_comparison_{model_type}router_{model_name}.pkl", "wb") as f:
        pickle.dump(fig, f)

    
    fig, axes = plt.subplots(ncols=3, sharey="row", figsize = (15, 5))
    plt.yscale("log")
    # axes[0].plot(np.mean(residuals_constant_jacobi, axis=1), label='Solver Only')
    # axes[0].plot(np.mean(residuals_hints, axis=1), label='HINTS')
    # axes[0].plot(np.mean(residuals_greedy, axis=1), label='Greedy')
    axes[0].plot(residuals_constant_jacobi[:, 20], label='Solver Only')
    axes[0].plot(residuals_hints[:, 20], label='HINTS')
    axes[0].plot(residuals_greedy[:, 20], label='Greedy')
    axes[0].set_title("Jacobi")
    axes[0].set_xlabel("Iterations")

    axes[1].plot(residuals_constant_gs[:, 20], label='Solver Only')
    axes[1].plot(residuals_hints_gs[:, 20], label='HINTS')
    axes[1].plot(residuals_greedy_gs[:, 20], label='Greedy')
    axes[1].plot(residuals_greedy_gs_jac[:, 20], label='Greedy-GS,Jacobi')
    axes[1].set_title("GS")
    axes[1].set_xlabel("Iterations")

    axes[2].plot(residuals_constant_mg[:, 20], label='Solver Only')
    axes[2].plot(residuals_hints_mg[:, 20], label='HINTS')
    axes[2].plot(residuals_greedy_mg[:, 20], label='Greedy')
    axes[2].set_title("MG")
    axes[2].set_xlabel("V cycles")

    line, label = axes[1].get_legend_handles_labels()
    fig.legend(line, label, loc="center right")

    plt.suptitle('Residual Comparison of Different Routing Strategies')

    plt.savefig(f"plots/separate_{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_residual_comparison_{model_type}router_{model_name}.png")
    with open(f"results/{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_residual_comparison_{model_type}router_{model_name}.pkl", "wb") as f:
        pickle.dump(fig, f)

    
    fig, axes = plt.subplots(ncols=3, nrows = 4, sharex = "col", sharey="row", figsize = (15, 20))

    # axes[0,0].plot(np.mean(mode_one_constant_jacobi, axis=1), label='Mode 1 Error')
    # axes[0,0].plot(np.mean(mode_five_constant_jacobi, axis=1), label='Mode 5 Error')
    # axes[0,0].plot(np.mean(mode_ten_constant_jacobi, axis=1), label='Mode 10 Error')
    axes[0,0].plot(mode_one_constant_jacobi[:, 20], label='Mode 1 Error')
    axes[0,0].plot(mode_five_constant_jacobi[:, 20], label='Mode 5 Error')
    axes[0,0].plot(mode_ten_constant_jacobi[:, 20], label='Mode 10 Error')
    axes[0, 0].set_title("Jacobi Only")
    axes[0, 0].set_xlabel("Iterations")

    axes[1, 0].plot(mode_one_hints[:, 20], label='Mode 1 Error')
    axes[1, 0].plot(mode_five_hints[:, 20], label='HINTS')
    axes[1, 0].plot(mode_ten_hints[:, 20], label='HINTS')
    axes[1, 0].set_title("HINTS-Jacobi")
    axes[1, 0].set_xlabel("Iterations")

    axes[2, 0].plot(mode_one_greedy[:, 20], label='Greedy')
    axes[2, 0].plot(mode_five_greedy[:, 20], label='Greedy')
    axes[2, 0].plot(mode_ten_greedy[:, 20], label='Greedy')
    axes[2, 0].set_title("Greedy-Jacobi")
    axes[2, 0].set_xlabel("Iterations")

    axes[0, 1].plot(mode_one_constant_gs[:, 20], label='Solver Only')
    axes[0, 1].plot(mode_five_constant_gs[:, 20], label='Solver Only')
    axes[0, 1].plot(mode_ten_constant_gs[:, 20], label='Solver Only')
    axes[0, 1].set_title("GS Only")
    axes[0, 1].set_xlabel("Iterations")

    axes[1, 1].plot(mode_one_hints_gs[:, 20], label='HINTS')
    axes[1, 1].plot(mode_five_hints_gs[:, 20], label='HINTS')
    axes[1, 1].plot(mode_ten_hints_gs[:, 20], label='HINTS')
    axes[1, 1].set_title("HINTS-GS")
    axes[1, 1].set_xlabel("Iterations")

    axes[2, 1].plot(mode_one_greedy_gs[:, 20], label='Greedy')
    axes[2, 1].plot(mode_five_greedy_gs[:, 20], label='Greedy')
    axes[2, 1].plot(mode_ten_greedy_gs[:, 20], label='Greedy')
    axes[2,1].set_title("Greedy-GS")
    axes[2, 1].set_xlabel("Iterations")

    axes[0,2].plot(mode_one_constant_mg[:, 20], label='Solver Only')
    axes[0,2].plot(mode_five_constant_mg[:, 20], label='Solver Only')
    axes[0,2].plot(mode_ten_constant_mg[:, 20], label='Solver Only')
    axes[0, 2].set_title("MG Only")
    axes[0, 2].set_xlabel("V cycles")

    axes[1,2].plot(mode_one_hints_mg[:, 20], label='HINTS')
    axes[1,2].plot(mode_five_hints_mg[:, 20], label='HINTS')
    axes[1,2].plot(mode_ten_hints_mg[:, 20], label='HINTS')
    axes[1, 2].set_title("HINTS-MG")
    axes[1, 2].set_xlabel("V cycles")

    axes[2,2].plot(mode_one_greedy_mg[:, 20], label='Greedy')
    axes[2,2].plot(mode_five_greedy_mg[:, 20], label='Greedy')
    axes[2,2].plot(mode_ten_greedy_mg[:, 20], label='Greedy')
    axes[2,2].set_title("Greedy-MG")
    axes[2,2].set_xlabel("V cycles")

    axes[3, 0].set_visible(False)
    axes[3, 1].plot(mode_one_greedy_gs_jac[:, 20], label='Greedy-GS,Jacobi')
    axes[3, 1].plot(mode_five_greedy_gs_jac[:, 20], label='Greedy-GS,Jacobi')
    axes[3, 1].plot(mode_ten_greedy_gs_jac[:, 20], label='Greedy-GS,Jacobi')
    axes[3,1].set_title("Greedy-GS,Jacobi")
    axes[3, 1].set_xlabel("Iteration")
    axes[3, 2].set_visible(False)

    for ax in axes:
        for a in ax:
            a.set_yscale("log")
    line, label = axes[0,0].get_legend_handles_labels()
    fig.legend(line, label, loc="center right")

    plt.suptitle('Mode-wise Error Comparison of Different Routing Strategies')
    plt.savefig(f"plots/separate_{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_modeerror_comparison_{model_type}router_{model_name}.png")
    with open(f"results/{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_modeerror_comparison_{model_type}router_{model_name}.pkl", "wb") as f:
        pickle.dump(fig, f)



    plt.figure(figsize=(10, 5))
    plt.plot(np.mean(residuals_greedy_mg, axis=1), label='Greedy-MG')
    plt.plot(np.mean(residuals_constant_mg, axis=1), label='MG Only')
    plt.plot(np.mean(residuals_hints_mg, axis=1), label='HINTS-MG')
    # plt.plot(np.mean(errors_true_greedy, axis=1), label='True Greedy Router')
    plt.xlabel('V Cycles')
    plt.ylabel('Residuals')
    plt.yscale("log")
    plt.legend()
    plt.title('Residual Comparison of Different Routing Strategies')
    plt.savefig(f"plots/MG_{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_residual_comparison_{model_type}router_{model_name}.png")

    auc_greedy = np.trapezoid(errors_greedy, axis=0)
    auc_greedy_gs = np.trapezoid(errors_greedy_gs, axis=0)
    auc_greedy_mg = np.trapezoid(errors_greedy_mg, axis=0)
    auc_greedy_gs_jac = np.trapezoid(errors_greedy_gs_jac, axis=0)
    auc_true_greedy = np.trapezoid(errors_true_greedy, axis=0)
    auc_constant_jacobi = np.trapezoid(errors_constant_jacobi, axis=0)
    auc_constant_gs = np.trapezoid(errors_constant_gs, axis=0)
    auc_constant_mg = np.trapezoid(errors_constant_mg, axis=0)
    auc_hints = np.trapezoid(errors_hints, axis=0)
    auc_hints_gs = np.trapezoid(errors_hints_gs, axis=0)
    auc_hints_mg = np.trapezoid(errors_hints_mg, axis=0)

    if os.path.exists(f"results/separate_{ml_model_type}_{ml_model_name}_error_comparison_{model_type}router_{model_name}.csv"):
        df_error = pd.read_csv(f"results/separate_{ml_model_type}_{ml_model_name}_error_comparison_{model_type}router_{model_name}.csv")
    else:
        df_error = pd.DataFrame({"Methods": ["Jacobi Only", "HINTS-Jacobi", "Greedy-Jacobi", "GS Only", "HINTS-GS", "Greedy-GS", "Greedy-GS,Jacobi", "MG Only", "HINTS-MG", "Greedy-MG"]})
        df_error["FinalError_1d_Poisson"] = ""
        df_error["AUC_Error_1d_Poisson"] = ""

        df_error["FinalError_2d_Poisson"] = ""
        df_error["AUC_Error_2d_Poisson"] = ""
        
        df_error["FinalError_1d_Helmholtz"] = ""
        df_error["AUC_Error_1d_Helmholtz"] = ""

        df_error["FinalError_2d_Helmholtz"] = ""
        df_error["AUC_Error_2d_Helmholtz"] = ""
    
    df_error[f"Mean_FinalError_{dim}d_{equation}"] = [np.mean(errors_constant_jacobi[-1])*(10**3), np.mean(errors_hints[-1])*(10**3), np.mean(errors_greedy[-1])*(10**3), np.mean(errors_constant_gs[-1])*(10**3), np.mean(errors_hints_gs[-1])*(10**3), np.mean(errors_greedy_gs[-1])*(10**3), np.mean(errors_greedy_gs_jac[-1])*(10**3), np.mean(errors_constant_mg[-1])*(10**3), np.mean(errors_hints_mg[-1])*(10**3), np.mean(errors_greedy_mg[-1])*(10**3)]
    df_error[f"Std_FinalError_{dim}d_{equation}"] = [np.std(errors_constant_jacobi[-1])*(10**3), np.std(errors_hints[-1])*(10**3), np.std(errors_greedy[-1])*(10**3), np.std(errors_constant_gs[-1])*(10**3), np.std(errors_hints_gs[-1])*(10**3), np.std(errors_greedy_gs[-1])*(10**3), np.std(errors_greedy_gs_jac[-1])*(10**3), np.std(errors_constant_mg[-1])*(10**3), np.std(errors_hints_mg[-1])*(10**3), np.std(errors_greedy_mg[-1])*(10**3)]
    df_error[f"FinalError_{dim}d_{equation}"] = df_error[f"Mean_FinalError_{dim}d_{equation}"].round(3).astype(str).str.cat(df_error[f"Std_FinalError_{dim}d_{equation}"].round(3).astype(str), sep = " (") + ")"


    df_error[f"Mean_AUC_Error_{dim}d_{equation}"] = [np.mean(auc_constant_jacobi)*(10**3), np.mean(auc_hints)*(10**3), np.mean(auc_greedy)*(10**3), np.mean(auc_constant_gs)*(10**3), np.mean(auc_hints_gs)*(10**3), np.mean(auc_greedy_gs)*(10**3), np.mean(auc_greedy_gs_jac)*(10**3), np.mean(auc_constant_mg)*(10**3), np.mean(auc_hints_mg)*(10**3), np.mean(auc_greedy_mg)*(10**3)]
    df_error[f"Std_AUC_Error_{dim}d_{equation}"] = [np.std(auc_constant_jacobi)*(10**3), np.std(auc_hints)*(10**3), np.std(auc_greedy)*(10**3), np.std(auc_constant_gs)*(10**3), np.std(auc_hints_gs)*(10**3), np.std(auc_greedy_gs)*(10**3), np.std(auc_greedy_gs_jac)*(10**3), np.std(auc_constant_mg)*(10**3), np.std(auc_hints_mg)*(10**3), np.std(auc_greedy_mg)*(10**3)]
    df_error[f"AUC_Error_{dim}d_{equation}"] = df_error[f"Mean_AUC_Error_{dim}d_{equation}"].round(3).astype(str).str.cat(df_error[f"Std_AUC_Error_{dim}d_{equation}"].round(3).astype(str), sep = " (") + ")"
    
    
    df_error.set_index("Methods", inplace = True)
    df_error.to_csv(f"results/separate_{ml_model_type}_{ml_model_name}_error_comparison_{model_type}router_{model_name}.csv")
    print("ERROR COMPARISON")
    print(df_error.to_latex(columns = ["FinalError_1d_Poisson", "AUC_Error_1d_Poisson", "FinalError_2d_Poisson", "AUC_Error_2d_Poisson", "FinalError_1d_Helmholtz", "AUC_Error_1d_Helmholtz", "FinalError_2d_Helmholtz", "AUC_Error_2d_Helmholtz"]))


    auc_greedy = np.trapezoid(residuals_greedy, axis=0)
    auc_greedy_gs = np.trapezoid(residuals_greedy_gs, axis=0)
    auc_greedy_mg = np.trapezoid(residuals_greedy_mg, axis=0)
    auc_greedy_gs_jac = np.trapezoid(residuals_greedy_gs_jac, axis=0)
    auc_constant_jacobi = np.trapezoid(residuals_constant_jacobi, axis=0)
    auc_constant_gs = np.trapezoid(residuals_constant_gs, axis=0)
    auc_constant_mg = np.trapezoid(residuals_constant_mg, axis=0)
    auc_hints = np.trapezoid(residuals_hints, axis=0)
    auc_hints_gs = np.trapezoid(residuals_hints_gs, axis=0)
    auc_hints_mg = np.trapezoid(residuals_hints_mg, axis=0)

    if os.path.exists(f"results/separate_{ml_model_type}_{ml_model_name}_residual_comparison_{model_type}router_{model_name}.csv"):
        df_residual = pd.read_csv(f"results/separate_{ml_model_type}_{ml_model_name}_residual_comparison_{model_type}router_{model_name}.csv")
    else:
        df_residual = pd.DataFrame({"Methods": ["Jacobi Only", "HINTS-Jacobi", "Greedy-Jacobi", "GS Only", "HINTS-GS", "Greedy-GS", "Greedy-GS,Jacobi", "MG Only", "HINTS-MG", "Greedy-MG"]})
        df_residual["FinalResidual_1d_Poisson"] = ""
        df_residual["AUC_Residual_1d_Poisson"] = ""

        df_residual["FinalResidual_2d_Poisson"] = ""
        df_residual["AUC_Residual_2d_Poisson"] = ""

        df_residual["FinalResidual_1d_Helmholtz"] = ""
        df_residual["AUC_Residual_1d_Helmholtz"] = ""

        df_residual["FinalResidual_2d_Helmholtz"] = ""
        df_residual["AUC_Residual_2d_Helmholtz"] = ""

    df_residual[f"Mean_FinalResidual_{dim}d_{equation}"] = [np.mean(residuals_constant_jacobi[-1])*(10**3), np.mean(residuals_hints[-1])*(10**3), np.mean(residuals_greedy[-1])*(10**3), np.mean(residuals_constant_gs[-1])*(10**3), np.mean(residuals_hints_gs[-1])*(10**3), np.mean(residuals_greedy_gs[-1])*(10**3), np.mean(residuals_greedy_gs_jac[-1])*(10**3), np.mean(residuals_constant_mg[-1])*(10**3), np.mean(residuals_hints_mg[-1])*(10**3), np.mean(residuals_greedy_mg[-1])*(10**3)]
    df_residual[f"Std_FinalResidual_{dim}d_{equation}"] = [np.std(residuals_constant_jacobi[-1])*(10**3), np.std(residuals_hints[-1])*(10**3), np.std(residuals_greedy[-1])*(10**3), np.std(residuals_constant_gs[-1])*(10**3), np.std(residuals_hints_gs[-1])*(10**3), np.std(residuals_greedy_gs[-1])*(10**3), np.std(residuals_greedy_gs_jac[-1])*(10**3), np.std(residuals_constant_mg[-1])*(10**3), np.std(residuals_hints_mg[-1])*(10**3), np.std(residuals_greedy_mg[-1])*(10**3)]

    df_residual[f"FinalResidual_{dim}d_{equation}"] = df_residual[f"Mean_FinalResidual_{dim}d_{equation}"].round(3).astype(str).str.cat(df_residual[f"Std_FinalResidual_{dim}d_{equation}"].round(3).astype(str), sep = " (") + ")"
    
    df_residual[f"Mean_AUC_Residual_{dim}d_{equation}"] = [np.mean(auc_constant_jacobi)*(10**3), np.mean(auc_hints)*(10**3), np.mean(auc_greedy)*(10**3), np.mean(auc_constant_gs)*(10**3), np.mean(auc_hints_gs)*(10**3), np.mean(auc_greedy_gs)*(10**3), np.mean(auc_greedy_gs_jac)*(10**3), np.mean(auc_constant_mg)*(10**3), np.mean(auc_hints_mg)*(10**3), np.mean(auc_greedy_mg)*(10**3)]
    df_residual[f"Std_AUC_Residual_{dim}d_{equation}"] = [np.std(auc_constant_jacobi)*(10**3), np.std(auc_hints)*(10**3), np.std(auc_greedy)*(10**3), np.std(auc_constant_gs)*(10**3), np.std(auc_hints_gs)*(10**3), np.std(auc_greedy_gs)*(10**3), np.std(auc_greedy_gs_jac)*(10**3), np.std(auc_constant_mg)*(10**3), np.std(auc_hints_mg)*(10**3), np.std(auc_greedy_mg)*(10**3)]
    df_residual[f"AUC_Residual_{dim}d_{equation}"] = df_residual[f"Mean_AUC_Residual_{dim}d_{equation}"].round(3).astype(str).str.cat(df_residual[f"Std_AUC_Residual_{dim}d_{equation}"].round(3).astype(str), sep = " (") + ")"
    df_residual.set_index("Methods", inplace = True)
    df_residual.to_csv(f"results/separate_{ml_model_type}_{ml_model_name}_residual_comparison_{model_type}router_{model_name}.csv")
    print("RESIDUAL COMPARISON")
    print(df_residual.to_latex(columns = ["FinalResidual_1d_Poisson", "AUC_Residual_1d_Poisson", "FinalResidual_2d_Poisson", "AUC_Residual_2d_Poisson", "FinalResidual_1d_Helmholtz", "AUC_Residual_1d_Helmholtz", "FinalResidual_2d_Helmholtz", "AUC_Residual_2d_Helmholtz"]))

  

    modeoneauc_greedy = np.trapezoid(mode_one_greedy, axis=0)
    modeoneauc_greedy_gs = np.trapezoid(mode_one_greedy_gs, axis=0)
    modeoneauc_greedy_mg = np.trapezoid(mode_one_greedy_mg, axis=0)
    modeoneauc_greedy_gs_jac = np.trapezoid(mode_one_greedy_gs_jac, axis=0)
    modeoneauc_constant_jacobi = np.trapezoid(mode_one_constant_jacobi, axis=0)
    modeoneauc_constant_gs = np.trapezoid(mode_one_constant_gs, axis=0)
    modeoneauc_constant_mg = np.trapezoid(mode_one_constant_mg, axis=0)
    modeoneauc_hints = np.trapezoid(mode_one_hints, axis=0)
    modeoneauc_hints_gs = np.trapezoid(mode_one_hints_gs, axis=0)
    modeoneauc_hints_mg = np.trapezoid(mode_one_hints_mg, axis=0)

    modefiveauc_greedy = np.trapezoid(mode_five_greedy, axis=0)
    modefiveauc_greedy_gs = np.trapezoid(mode_five_greedy_gs, axis=0)
    modefiveauc_greedy_mg = np.trapezoid(mode_five_greedy_mg, axis=0)
    modefiveauc_greedy_gs_jac = np.trapezoid(mode_five_greedy_gs_jac, axis=0)
    modefiveauc_constant_jacobi = np.trapezoid(mode_five_constant_jacobi, axis=0)
    modefiveauc_constant_gs = np.trapezoid(mode_five_constant_gs, axis=0)
    modefiveauc_constant_mg = np.trapezoid(mode_five_constant_mg, axis=0)
    modefiveauc_hints = np.trapezoid(mode_five_hints, axis=0)
    modefiveauc_hints_gs = np.trapezoid(mode_five_hints_gs, axis=0)
    modefiveauc_hints_mg = np.trapezoid(mode_five_hints_mg, axis=0)

    modetenauc_greedy = np.trapezoid(mode_ten_greedy, axis=0)
    modetenauc_greedy_gs = np.trapezoid(mode_ten_greedy_gs, axis=0)
    modetenauc_greedy_mg = np.trapezoid(mode_ten_greedy_mg, axis=0)
    modetenauc_greedy_gs_jac = np.trapezoid(mode_ten_greedy_gs_jac, axis=0)
    modetenauc_constant_jacobi = np.trapezoid(mode_ten_constant_jacobi, axis=0)
    modetenauc_constant_gs = np.trapezoid(mode_ten_constant_gs, axis=0)
    modetenauc_constant_mg = np.trapezoid(mode_ten_constant_mg, axis=0)
    modetenauc_hints = np.trapezoid(mode_ten_hints, axis=0)
    modetenauc_hints_gs = np.trapezoid(mode_ten_hints_gs, axis=0)
    modetenauc_hints_mg = np.trapezoid(mode_ten_hints_mg, axis=0)


    if os.path.exists(f"results/separate_{ml_model_type}_{ml_model_name}_modes_comparison_{model_type}router_{model_name}.csv"):
        df_mode_error = pd.read_csv(f"results/separate_{ml_model_type}_{ml_model_name}_modes_comparison_{model_type}router_{model_name}.csv")
    else:
        df_mode_error = pd.DataFrame({"Methods": ["Jacobi Only", "HINTS-Jacobi", "Greedy-Jacobi", "GS Only", "HINTS-GS", "Greedy-GS", "Greedy-GS,Jacobi", "MG Only", "HINTS-MG", "Greedy-MG"]})
        df_mode_error["Final_Mode1_1d_Poisson"] = ""
        df_mode_error["AUC_Mode1_1d_Poisson"] = ""
        df_mode_error["Final_Mode5_1d_Poisson"] = ""
        df_mode_error["AUC_Mode5_1d_Poisson"] = ""
        df_mode_error["Final_Mode10_1d_Poisson"] = ""
        df_mode_error["AUC_Mode10_1d_Poisson"] = ""

        df_mode_error["Final_Mode1_2d_Poisson"] = ""
        df_mode_error["AUC_Mode1_2d_Poisson"] = ""
        df_mode_error["Final_Mode5_2d_Poisson"] = ""
        df_mode_error["AUC_Mode5_2d_Poisson"] = ""
        df_mode_error["Final_Mode10_2d_Poisson"] = ""
        df_mode_error["AUC_Mode10_2d_Poisson"] = ""

        df_mode_error["Final_Mode1_1d_Helmholtz"] = ""
        df_mode_error["AUC_Mode1_1d_Helmholtz"] = ""
        df_mode_error["Final_Mode5_1d_Helmholtz"] = ""
        df_mode_error["AUC_Mode5_1d_Helmholtz"] = ""
        df_mode_error["Final_Mode10_1d_Helmholtz"] = ""
        df_mode_error["AUC_Mode10_1d_Helmholtz"] = ""

        df_mode_error["Final_Mode1_2d_Helmholtz"] = ""
        df_mode_error["AUC_Mode1_2d_Helmholtz"] = ""
        df_mode_error["Final_Mode5_2d_Helmholtz"] = ""
        df_mode_error["AUC_Mode5_2d_Helmholtz"] = ""
        df_mode_error["Final_Mode10_2d_Helmholtz"] = ""
        df_mode_error["AUC_Mode10_2d_Helmholtz"] = ""

    df_mode_error[f"Mean_Final_Mode1_{dim}d_{equation}"] = [np.mean(mode_one_constant_jacobi[-1])*(10**3), np.mean(mode_one_hints[-1])*(10**3), np.mean(mode_one_greedy[-1])*(10**3), np.mean(mode_one_constant_gs[-1])*(10**3), np.mean(mode_one_hints_gs[-1])*(10**3), np.mean(mode_one_greedy_gs[-1])*(10**3), np.mean(mode_one_greedy_gs_jac[-1])*(10**3), np.mean(mode_one_constant_mg[-1])*(10**3), np.mean(mode_one_hints_mg[-1])*(10**3), np.mean(mode_one_greedy_mg[-1])*(10**3)]
    df_mode_error[f"Std_Final_Mode1_{dim}d_{equation}"] = [np.std(mode_one_constant_jacobi[-1])*(10**3), np.std(mode_one_hints[-1])*(10**3), np.std(mode_one_greedy[-1])*(10**3), np.std(mode_one_constant_gs[-1])*(10**3), np.std(mode_one_hints_gs[-1])*(10**3), np.std(mode_one_greedy_gs[-1])*(10**3), np.std(mode_one_greedy_gs_jac[-1])*(10**3), np.std(mode_one_constant_mg[-1])*(10**3), np.std(mode_one_hints_mg[-1])*(10**3), np.std(mode_one_greedy_mg[-1])*(10**3)]
    df_mode_error[f"Final_Mode1_{dim}d_{equation}"] = df_mode_error[f"Mean_Final_Mode1_{dim}d_{equation}"].round(3).astype(str).str.cat(df_mode_error[f"Std_Final_Mode1_{dim}d_{equation}"].round(3).astype(str), sep = " (") + ")"

    df_mode_error[f"Mean_AUC_Mode1_{dim}d_{equation}"] = [np.mean(modeoneauc_constant_jacobi)*(10**3), np.mean(modeoneauc_hints)*(10**3), np.mean(modeoneauc_greedy)*(10**3), np.mean(modeoneauc_constant_gs)*(10**3), np.mean(modeoneauc_hints_gs)*(10**3), np.mean(modeoneauc_greedy_gs)*(10**3), np.mean(modeoneauc_greedy_gs_jac)*(10**3), np.mean(modeoneauc_constant_mg)*(10**3), np.mean(modeoneauc_hints_mg)*(10**3), np.mean(modeoneauc_greedy_mg)*(10**3)]
    df_mode_error[f"Std_AUC_Mode1_{dim}d_{equation}"] = [np.std(modeoneauc_constant_jacobi)*(10**3), np.std(modeoneauc_hints)*(10**3), np.std(modeoneauc_greedy)*(10**3), np.std(modeoneauc_constant_gs)*(10**3), np.std(modeoneauc_hints_gs)*(10**3), np.std(modeoneauc_greedy_gs)*(10**3), np.std(modeoneauc_greedy_gs_jac)*(10**3), np.std(modeoneauc_constant_mg)*(10**3), np.std(modeoneauc_hints_mg)*(10**3), np.std(modeoneauc_greedy_mg)*(10**3)]
    df_mode_error[f"AUC_Mode1_{dim}d_{equation}"] = df_mode_error[f"Mean_AUC_Mode1_{dim}d_{equation}"].round(3).astype(str).str.cat(df_mode_error[f"Std_AUC_Mode1_{dim}d_{equation}"].round(3).astype(str), sep = " (") + ")"

    df_mode_error[f"Mean_Final_Mode5_{dim}d_{equation}"] = [np.mean(mode_five_constant_jacobi[-1])*(10**3), np.mean(mode_five_hints[-1])*(10**3), np.mean(mode_five_greedy[-1])*(10**3), np.mean(mode_five_constant_gs[-1])*(10**3), np.mean(mode_five_hints_gs[-1])*(10**3), np.mean(mode_five_greedy_gs[-1])*(10**3), np.mean(mode_five_greedy_gs_jac[-1])*(10**3), np.mean(mode_five_constant_mg[-1])*(10**3), np.mean(mode_five_hints_mg[-1])*(10**3), np.mean(mode_five_greedy_mg[-1])*(10**3)]
    df_mode_error[f"Std_Final_Mode5_{dim}d_{equation}"] = [np.std(mode_five_constant_jacobi[-1])*(10**3), np.std(mode_five_hints[-1])*(10**3), np.std(mode_five_greedy[-1])*(10**3), np.std(mode_five_constant_gs[-1])*(10**3), np.std(mode_five_hints_gs[-1])*(10**3), np.std(mode_five_greedy_gs[-1])*(10**3), np.std(mode_five_greedy_gs_jac[-1])*(10**3), np.std(mode_five_constant_mg[-1])*(10**3), np.std(mode_five_hints_mg[-1])*(10**3), np.std(mode_five_greedy_mg[-1])*(10**3)]
    df_mode_error[f"Final_Mode5_{dim}d_{equation}"] = df_mode_error[f"Mean_Final_Mode5_{dim}d_{equation}"].round(3).astype(str).str.cat(df_mode_error[f"Std_Final_Mode5_{dim}d_{equation}"].round(3).astype(str), sep = " (") + ")"

    df_mode_error[f"Mean_AUC_Mode5_{dim}d_{equation}"] = [np.mean(modefiveauc_constant_jacobi)*(10**3), np.mean(modefiveauc_hints)*(10**3), np.mean(modefiveauc_greedy)*(10**3), np.mean(modefiveauc_constant_gs)*(10**3), np.mean(modefiveauc_hints_gs)*(10**3), np.mean(modefiveauc_greedy_gs)*(10**3), np.mean(modefiveauc_greedy_gs_jac)*(10**3), np.mean(modefiveauc_constant_mg)*(10**3), np.mean(modefiveauc_hints_mg)*(10**3), np.mean(modefiveauc_greedy_mg)*(10**3)]
    df_mode_error[f"Std_AUC_Mode5_{dim}d_{equation}"] = [np.std(modefiveauc_constant_jacobi)*(10**3), np.std(modefiveauc_hints)*(10**3), np.std(modefiveauc_greedy)*(10**3), np.std(modefiveauc_constant_gs)*(10**3), np.std(modefiveauc_hints_gs)*(10**3), np.std(modefiveauc_greedy_gs)*(10**3), np.std(modefiveauc_greedy_gs_jac)*(10**3), np.std(modefiveauc_constant_mg)*(10**3), np.std(modefiveauc_hints_mg)*(10**3), np.std(modefiveauc_greedy_mg)*(10**3)]
    df_mode_error[f"AUC_Mode5_{dim}d_{equation}"] = df_mode_error[f"Mean_AUC_Mode5_{dim}d_{equation}"].round(3).astype(str).str.cat(df_mode_error[f"Std_AUC_Mode5_{dim}d_{equation}"].round(3).astype(str), sep = " (") + ")"

    df_mode_error[f"Mean_Final_Mode10_{dim}d_{equation}"] = [np.mean(mode_ten_constant_jacobi[-1])*(10**3), np.mean(mode_ten_hints[-1])*(10**3), np.mean(mode_ten_greedy[-1])*(10**3), np.mean(mode_ten_constant_gs[-1])*(10**3), np.mean(mode_ten_hints_gs[-1])*(10**3), np.mean(mode_ten_greedy_gs[-1])*(10**3), np.mean(mode_ten_greedy_gs_jac[-1])*(10**3), np.mean(mode_ten_constant_mg[-1])*(10**3), np.mean(mode_ten_hints_mg[-1])*(10**3), np.mean(mode_ten_greedy_mg[-1])*(10**3)]
    df_mode_error[f"Std_Final_Mode10_{dim}d_{equation}"] = [np.std(mode_ten_constant_jacobi[-1])*(10**3), np.std(mode_ten_hints[-1])*(10**3), np.std(mode_ten_greedy[-1])*(10**3), np.std(mode_ten_constant_gs[-1])*(10**3), np.std(mode_ten_hints_gs[-1])*(10**3), np.std(mode_ten_greedy_gs[-1])*(10**3), np.std(mode_ten_greedy_gs_jac[-1])*(10**3), np.std(mode_ten_constant_mg[-1])*(10**3), np.std(mode_ten_hints_mg[-1])*(10**3), np.std(mode_ten_greedy_mg[-1])*(10**3)]
    df_mode_error[f"Final_Mode10_{dim}d_{equation}"] = df_mode_error[f"Mean_Final_Mode10_{dim}d_{equation}"].round(3).astype(str).str.cat(df_mode_error[f"Std_Final_Mode10_{dim}d_{equation}"].round(3).astype(str), sep = " (") + ")"

    df_mode_error[f"Mean_AUC_Mode10_{dim}d_{equation}"] = [np.mean(modetenauc_constant_jacobi)*(10**3), np.mean(modetenauc_hints)*(10**3), np.mean(modetenauc_greedy)*(10**3), np.mean(modetenauc_constant_gs)*(10**3), np.mean(modetenauc_hints_gs)*(10**3), np.mean(modetenauc_greedy_gs)*(10**3), np.mean(modetenauc_greedy_gs_jac)*(10**3), np.mean(modetenauc_constant_mg)*(10**3), np.mean(modetenauc_hints_mg)*(10**3), np.mean(modetenauc_greedy_mg)*(10**3)]
    df_mode_error[f"Std_AUC_Mode10_{dim}d_{equation}"] = [np.std(modetenauc_constant_jacobi)*(10**3), np.std(modetenauc_hints)*(10**3), np.std(modetenauc_greedy)*(10**3), np.std(modetenauc_constant_gs)*(10**3), np.std(modetenauc_hints_gs)*(10**3), np.std(modetenauc_greedy_gs)*(10**3), np.std(modetenauc_greedy_gs_jac)*(10**3), np.std(modetenauc_constant_mg)*(10**3), np.std(modetenauc_hints_mg)*(10**3), np.std(modetenauc_greedy_mg)*(10**3)]
    df_mode_error[f"AUC_Mode10_{dim}d_{equation}"] = df_mode_error[f"Mean_AUC_Mode10_{dim}d_{equation}"].round(3).astype(str).str.cat(df_mode_error[f"Std_AUC_Mode10_{dim}d_{equation}"].round(3).astype(str), sep = " (") + ")"
    df_mode_error.set_index("Methods", inplace = True)

    df_mode_error.to_csv(f"results/separate_{ml_model_type}_{ml_model_name}_modes_comparison_{model_type}router_{model_name}.csv")
    print("Poisson 1d Mode COMPARISON")
    print(df_mode_error.to_latex(columns = ["Final_Mode1_1d_Poisson", "AUC_Mode1_1d_Poisson", "Final_Mode5_1d_Poisson", "AUC_Mode5_1d_Poisson", "Final_Mode10_1d_Poisson", "AUC_Mode10_1d_Poisson"]))


    print("Poisson 2d mode error COMPARISON")
    print(df_mode_error.to_latex(columns = ["Final_Mode1_2d_Poisson", "AUC_Mode1_2d_Poisson", "Final_Mode5_2d_Poisson", "AUC_Mode5_2d_Poisson", "Final_Mode10_2d_Poisson", "AUC_Mode10_2d_Poisson"]))

    print("Helmholtz 1d mode error COMPARISON")
    print(df_mode_error.to_latex(columns = ["Final_Mode1_1d_Helmholtz", "AUC_Mode1_1d_Helmholtz", "Final_Mode5_1d_Helmholtz", "AUC_Mode5_1d_Helmholtz", "Final_Mode10_1d_Helmholtz", "AUC_Mode10_1d_Helmholtz"]))

    print("Helmholtz 2d mode error COMPARISON")
    print(df_mode_error.to_latex(columns = ["Final_Mode1_2d_Helmholtz", "AUC_Mode1_2d_Helmholtz", "Final_Mode5_2d_Helmholtz", "AUC_Mode5_2d_Helmholtz", "Final_Mode10_2d_Helmholtz", "AUC_Mode10_2d_Helmholtz"]))


def run_2(model_type, ml_model_type, n_test, dim, boundary, equation, ckp_dir, model_name, ml_model_name, data_dir, in_channels, device, extra = 1000):
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    ml_ckp_path = ckp_dir + f"/{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_best.pth"
    ml_args_path = ckp_dir + f"/{ml_model_type}_{ml_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_args.json"

    args_path = ckp_dir + f"/{model_type}router_{model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_jacobi_0.5args.json"

    if os.path.exists(args_path):
        print(f"Loading training arguments from {args_path}...")
        with open(args_path, "r") as f:
            arguments = json.load(f)
    
    if os.path.exists(ml_args_path):
        with open(ml_args_path, "r") as f:
            ml_arguments = json.load(f)
    else:
        raise ValueError("ML Args Path Not found")
    # Creating/Loading Data
    print("Creating Data")

    test_or_train = "test"

    if os.path.exists(f"{data_dir}/router_{test_or_train}_data_{equation}_{boundary}_{dim}d_{in_channels}c_{n_test}s.pt"):
        print(f"Loading data from {data_dir}...")
        with open(f"{data_dir}/router_{test_or_train}_data_{equation}_{boundary}_{dim}d_{in_channels}c_{n_test}s.pt", "rb") as f:
            test_data = torch.load(f)
    else:
        with open(f"args/grf_args.json", "r") as f:
            arguments_grf = json.load(f)

        grf = GaussianRandomField(num_samples=arguments["N"],
                                    dim=dim,
                                    alpha=arguments_grf["alpha"],
                                    beta=arguments_grf["beta"],
                                    gamma=arguments_grf["gamma"],
                                    device=device,
                                    seed = 2134)
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
        test_data = []
        for i in range(n_test + extra):
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
                    pde = HelmholtzEquation2D(a_func=a[i].flatten() if in_channels > 1 else a, f_func = f[i].flatten(), k2=k2[i].flatten(),boundary=boundary, x=x, y=y, device=device)
                new_shape = (arguments["N"], arguments["N"]) 
                u_sol = torch.tensor(pde.u.reshape(new_shape), dtype=torch.float32, device=device)
                u_sol = u_sol - torch.mean(u_sol) if equation == "Poisson" else u_sol
            
            if in_channels > 1:
                if equation == "Poisson":
                    input = torch.concatenate((a[i, None, :], f[i, None, :]), dim=0)
                else:
                    input = torch.concatenate((a[i, None, :], k2[i, None, :], f[i, None, :]), dim=0)                        
            else:
                if equation == "Poisson":
                    input = f[i, None, :]
                else:
                    input = torch.concatenate((k2[i, None, :], f[i, None, :]), dim=0)
            residual = pde.compute_residual(u_sol.flatten())
            if torch.linalg.norm(residual) > 1:
                continue
            test_data.append((input, u_sol))
            if len(test_data) == n_test:
                break
        if len(test_data) < n_test:
            print(f"Generated {len(test_data)} test samples")
            raise ValueError("Not enough data generated. Try increasing the extra variable.")
        with open(f"{data_dir}/router_test_data_{equation}_{boundary}_{dim}d_{in_channels}c_{arguments["n_train"]}s.pt", "wb") as f:
            torch.save(test_data, f)
    print("Data creation/loading completed.")
    print(f"Test data size: {len(test_data)}")
    print(f"Size of each input: {test_data[0][0].shape}, Size of each solution: {test_data[0][1].shape}")
    # Change this later 
    test_dataset = PDEDataset(test_data)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=arguments["batch_size"], shuffle=True)
    # test_dataset = PDEDataset(test_data[:2])
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False)
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
    

    ckp_path = ckp_dir + f"/{model_type}router_{model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_jacobi_0.5_best.pth"
    save_path = ckp_dir + f"/{model_type}router_{model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_jacobi_0.5"
    args_path = ckp_dir + f"/{model_type}router_{model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_jacobi_0.5args.json"

    if os.path.exists(args_path):
        print(f"Loading training arguments from {args_path}...")
        with open(args_path, "r") as f:
            arguments = json.load(f)
    else:
        raise ValueError("Router Args Path Not found")
    if model_type == "lstm":
        print(f"HERE")
        if dim == 1:
            router = LSTMGreedyRouter(None, ml_arguments["N"]*(new_in_channels + 1), arguments["hidden_dim"], arguments["num_layers"], 2, arguments["dropout"]).to(device)
        else:
            router = LSTMGreedyRouter(None, ml_arguments["N"]*ml_arguments["N"]*(new_in_channels + 1), arguments["hidden_dim"], arguments["num_layers"], 2, arguments["dropout"]).to(device)
        # router = LSTMGreedyRouter(None, ml_arguments["N"]*(in_channels + 1), arguments["hidden_dim"], arguments["num_layers"], num_solvers, arguments["dropout"]).to(device)
    ckp = None
    if os.path.exists(ckp_path):
        print(f"Loading model checkpoint from {ckp_path}...")
        ckp = torch.load(ckp_path, map_location=device)
    
    if ckp:
        print(f"Loading router state dict...")
        router.load_state_dict(ckp["model"])
        print(f"Loaded router state dict")
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

    model = HybridSolver(N=arguments["N"], dim=dim, in_channels=in_channels, boundary=boundary, equation=pde,
                                    suite_solver=[WeightedJacobiSolver(pde, device)]+[ml_model], router=router, tol=1e-7, max_iters=arguments["max_iters"], threshold=0.1).to(device)
    
    
    model.eval()
    loss = ApproxGreedyRouterLoss(centered=(equation == "Poisson"))
    errors_greedy, loss_greedy, residuals_greedy, mode_one_greedy, mode_five_greedy, mode_ten_greedy = test_model(model, test_loader, in_channels, dim,loss, equation == "Poisson")

    

    # gs = GaussSeidelSolver(pde, device)
    
    
    # ckp_path = ckp_dir + f"/{model_type}router_{model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_gs_best.pth"
    # args_path = ckp_dir + f"/{model_type}router_{model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_gsargs.json"
    # if os.path.exists(args_path):
    #     print(f"Loading training arguments from {args_path}...")
    #     with open(args_path, "r") as f:
    #         arguments = json.load(f)
    # else:
    #     raise ValueError("GS Router Args Path Not found")
    
    # if model_type == "lstm":
    #     print(f"HERE")
    #     if dim == 1:
    #         router_gs = LSTMGreedyRouter(None, ml_arguments["N"]*(new_in_channels + 1), arguments["hidden_dim"], arguments["num_layers"], 2, arguments["dropout"]).to(device)
    #     else:
    #         router_gs = LSTMGreedyRouter(None, ml_arguments["N"]*ml_arguments["N"]*(new_in_channels + 1), arguments["hidden_dim"], arguments["num_layers"], 2, arguments["dropout"]).to(device)

    # ckp = None

    # if os.path.exists(ckp_path):
    #     print(f"Loading GS model checkpoint from {ckp_path}...")
    #     ckp = torch.load(ckp_path, map_location=device)
    
    # if ckp:
    #     print(f"Loading GS router state dict...")
    #     router_gs.load_state_dict(ckp["model"])
    #     print(f"Loaded router state dict")

    # model_greedy_gs = HybridSolver(N=arguments["N"], dim=dim, in_channels=in_channels, boundary=boundary, equation=pde,
    #                                 suite_solver=[gs]+[ml_model], router=router_gs, tol=1e-7, max_iters=arguments["max_iters"], threshold=0.1).to(device)
    
    # model_greedy_gs.eval()
    # errors_greedy_gs, loss_greedy_gs, residuals_greedy_gs, mode_one_greedy_gs, mode_five_greedy_gs, mode_ten_greedy_gs = test_model(model_greedy_gs, test_loader, in_channels, dim, loss, equation == "Poisson")

    # temp_model_name = model_name
    # ckp_path = ckp_dir + f"/{model_type}router_{temp_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_gs,jacobi_best.pth"
    # args_path = ckp_dir + f"/{model_type}router_{temp_model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_gs,jacobiargs.json"
    # if os.path.exists(args_path):
    #     print(f"Loading training arguments from {args_path}...")
    #     with open(args_path, "r") as f:
    #         arguments = json.load(f)
    # else:
    #     raise ValueError("GS,Jacobi Router Args Path Not found")

    # if model_type == "lstm":
    #     print(f"HERE")
    #     if dim == 1:
    #         router_gs_jac = LSTMGreedyRouter(None, ml_arguments["N"]*(new_in_channels + 1), arguments["hidden_dim"], arguments["num_layers"], 3, arguments["dropout"]).to(device)
    #     else:
    #         router_gs_jac = LSTMGreedyRouter(None, ml_arguments["N"]*ml_arguments["N"]*(new_in_channels + 1), arguments["hidden_dim"], arguments["num_layers"], 3, arguments["dropout"]).to(device)
    # ckp = None

    # if os.path.exists(ckp_path):
    #     print(f"Loading GS,Jacobi model checkpoint from {ckp_path}...")
    #     ckp = torch.load(ckp_path, map_location=device)
    
    # if ckp:
    #     print(f"Loading GS,Jaocbi router state dict...")
    #     router_gs_jac.load_state_dict(ckp["model"])
    #     print(f"Loaded router state dict")

    # model_greedy_gs_jac = HybridSolver(N=arguments["N"], dim=dim, in_channels=in_channels, boundary=boundary, equation=pde,
    #                                 suite_solver=[gs]+[WeightedJacobiSolver(pde, device)] +[ml_model], router=router_gs_jac, tol=1e-7, max_iters=arguments["max_iters"], threshold=0.1).to(device)
    
    # model_greedy_gs_jac.eval()
    # errors_greedy_gs_jac, loss_greedy_gs_jac, residuals_greedy_gs_jac, mode_one_greedy_gs_jac, mode_five_greedy_gs_jac, mode_ten_greedy_gs_jac = test_model(model_greedy_gs_jac, test_loader, in_channels, dim, loss, equation == "Poisson")

    list_of_solvers_potential = ["jacobi_0.5", "jacobi_0.67", "jacobi_0.75","jacobi_0.8", "jacobi"]
    solverrs = [WeightedJacobiSolver(pde, device, 0.5), WeightedJacobiSolver(pde, device, 0.67),  WeightedJacobiSolver(pde, device, 0.75), WeightedJacobiSolver(pde, device, 0.8), WeightedJacobiSolver(pde, device)]# , WeightedJacobiSolver(pde, device, 0.5)]
    for i in range(1, 5):
        list_of_solvers_potential[i] = list_of_solvers_potential[i - 1] + "," + list_of_solvers_potential[i]

    result_error = []
    result_loss = []
    result_residual = []
    result_mode_one =[]
    result_mode_five = []
    result_mode_ten = []

    for i in range(5):
        ckp_path = ckp_dir + f"/{model_type}router_{model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_{list_of_solvers_potential[i]}_best.pth"
        args_path = ckp_dir + f"/{model_type}router_{model_name}_{equation}_{boundary}_{dim}d_{in_channels}c_{list_of_solvers_potential[i]}args.json"
        if os.path.exists(args_path):
            print(f"Loading training arguments from {args_path}...")
            with open(args_path, "r") as f:
                arguments = json.load(f)
        else:
            raise ValueError(f"{list_of_solvers_potential[i]} Router Args Path Not found")
        
        if model_type == "lstm":
            print(f"HERE")
            if dim == 1:
                router_gs = LSTMGreedyRouter(None, ml_arguments["N"]*(new_in_channels + 1), arguments["hidden_dim"], arguments["num_layers"], i+ 2, arguments["dropout"]).to(device)
            else:
                router_gs = LSTMGreedyRouter(None, ml_arguments["N"]*ml_arguments["N"]*(new_in_channels + 1), arguments["hidden_dim"], arguments["num_layers"], i + 2, arguments["dropout"]).to(device)

        ckp = None

        if os.path.exists(ckp_path):
            print(f"Loading GS model checkpoint from {ckp_path}...")
            ckp = torch.load(ckp_path, map_location=device)
        
        if ckp:
            print(f"Loading GS router state dict...")
            router_gs.load_state_dict(ckp["model"])
            print(f"Loaded router state dict")

        model_g = HybridSolver(N=arguments["N"], dim=dim, in_channels=in_channels, boundary=boundary, equation=pde,
                                        suite_solver=solverrs[:(i + 1)]+[ml_model], router=router_gs, tol=1e-7, max_iters=arguments["max_iters"], threshold=0.1).to(device)
        
        model_g.eval()
        errors_g, loss_g, residuals_g, mode_one_g, mode_five_g, mode_ten_g = test_model(model_g, test_loader, in_channels, dim, loss, equation == "Poisson")
        result_error.append(errors_g)
        result_loss.append(loss_g)
        result_residual.append(residuals_g)
        result_mode_one.append(mode_one_g)
        result_mode_five.append(mode_five_g)
        result_mode_ten.append(mode_ten_g)

    
 

    auc_greedy = np.trapezoid(errors_greedy, axis=0)
    # auc_greedy_gs = np.trapezoid(errors_greedy_gs, axis=0)
    # auc_greedy_gs_jac = np.trapezoid(errors_greedy_gs_jac, axis=0)
    auc_errors = []
    for i in range(5):
        auc_errors.append(np.trapezoid(result_error[i], axis=0))

    if os.path.exists(f"results/increasingK_2_separate_{ml_model_type}_{ml_model_name}_error_comparison_{model_type}router_{model_name}.csv"):
        df_error = pd.read_csv(f"results/increasingK_2_separate_{ml_model_type}_{ml_model_name}_error_comparison_{model_type}router_{model_name}.csv")
    else:
        df_error = pd.DataFrame({"Methods": ["1", "2", "3", "4", "5"]})# ["Greedy-Jacobi", "Greedy-GS", "2", "3", "4", "5"]})
        df_error["FinalError_1d_Poisson"] = ""
        df_error["AUC_Error_1d_Poisson"] = ""

        df_error["FinalError_2d_Poisson"] = ""
        df_error["AUC_Error_2d_Poisson"] = ""
        
        df_error["FinalError_1d_Helmholtz"] = ""
        df_error["AUC_Error_1d_Helmholtz"] = ""

        df_error["FinalError_2d_Helmholtz"] = ""
        df_error["AUC_Error_2d_Helmholtz"] = ""

    mean_final_error = []#[np.mean(errors_greedy[-1])*(10**3), np.mean(errors_greedy_gs[-1])*(10**3), np.mean(errors_greedy_gs_jac[-1])*(10**3)]
    std_final_error = []# [np.std(errors_greedy[-1])*(10**3), np.std(errors_greedy_gs[-1])*(10**3), np.std(errors_greedy_gs_jac[-1])*(10**3)]
    mean_auc_error = []# [np.mean(auc_greedy)*(10**3), np.mean(auc_greedy_gs)*(10**3), np.mean(auc_greedy_gs_jac)*(10**3)]
    std_auc_error = [] #[np.std(auc_greedy)*(10**3), np.std(auc_greedy_gs)*(10**3), np.std(auc_greedy_gs_jac)*(10**3)]
    for i in range(5):
        mean_final_error.append(np.mean(result_error[i][-1])*(10**3))
        std_final_error.append(np.std(result_error[i][-1])*(10**3))
        mean_auc_error.append(np.mean(auc_errors[i])*(10**3))
        std_auc_error.append(np.std(auc_errors[i])*(10**3))
    
    df_error[f"Mean_FinalError_{dim}d_{equation}"] =  mean_final_error
    df_error[f"Std_FinalError_{dim}d_{equation}"] = std_final_error
    df_error[f"FinalError_{dim}d_{equation}"] = df_error[f"Mean_FinalError_{dim}d_{equation}"].round(3).astype(str).str.cat(df_error[f"Std_FinalError_{dim}d_{equation}"].round(3).astype(str), sep = " (") + ")"


    df_error[f"Mean_AUC_Error_{dim}d_{equation}"] = mean_auc_error
    df_error[f"Std_AUC_Error_{dim}d_{equation}"] = std_auc_error
    df_error[f"AUC_Error_{dim}d_{equation}"] = df_error[f"Mean_AUC_Error_{dim}d_{equation}"].round(3).astype(str).str.cat(df_error[f"Std_AUC_Error_{dim}d_{equation}"].round(3).astype(str), sep = " (") + ")"
    
    
    df_error.set_index("Methods", inplace = True)
    df_error.to_csv(f"results/increasingK_2_separate_{ml_model_type}_{ml_model_name}_error_comparison_{model_type}router_{model_name}.csv")
    print("ERROR COMPARISON")
    print(df_error.to_latex(columns = ["FinalError_1d_Poisson", "AUC_Error_1d_Poisson", "FinalError_2d_Poisson", "AUC_Error_2d_Poisson", "FinalError_1d_Helmholtz", "AUC_Error_1d_Helmholtz", "FinalError_2d_Helmholtz", "AUC_Error_2d_Helmholtz"]))



    auc_greedy = np.trapezoid(residuals_greedy, axis=0)
    #auc_greedy_gs = np.trapezoid(residuals_greedy_gs, axis=0)
    #auc_greedy_gs_jac = np.trapezoid(residuals_greedy_gs_jac, axis=0)
    auc_errors = []
    for i in range(5):
        auc_errors.append(np.trapezoid(result_residual[i], axis=0))

    if os.path.exists(f"results/increasingK_2_separate_{ml_model_type}_{ml_model_name}_residual_comparison_{model_type}router_{model_name}.csv"):
        df_residual = pd.read_csv(f"results/increasingK_2_separate_{ml_model_type}_{ml_model_name}_residual_comparison_{model_type}router_{model_name}.csv")
    else:
        df_residual = pd.DataFrame({"Methods": ["1", "2", "3", "4", "5"]})#["Greedy-Jacobi", "Greedy-GS", "2", "3", "4", "5"]})
        df_residual["FinalResidual_1d_Poisson"] = ""
        df_residual["AUC_Residual_1d_Poisson"] = ""

        df_residual["FinalResidual_2d_Poisson"] = ""
        df_residual["AUC_Residual_2d_Poisson"] = ""

        df_residual["FinalResidual_1d_Helmholtz"] = ""
        df_residual["AUC_Residual_1d_Helmholtz"] = ""

        df_residual["FinalResidual_2d_Helmholtz"] = ""
        df_residual["AUC_Residual_2d_Helmholtz"] = ""

    mean_final_error = []#[np.mean(residuals_greedy[-1])*(10**3), np.mean(residuals_greedy_gs[-1])*(10**3), np.mean(residuals_greedy_gs_jac[-1])*(10**3)]
    std_final_error = []# [np.std(residuals_greedy[-1])*(10**3), np.std(residuals_greedy_gs[-1])*(10**3), np.std(residuals_greedy_gs_jac[-1])*(10**3)]
    mean_auc_error = [] #[np.mean(auc_greedy)*(10**3), np.mean(auc_greedy_gs)*(10**3), np.mean(auc_greedy_gs_jac)*(10**3)]
    std_auc_error = []# [np.std(auc_greedy)*(10**3), np.std(auc_greedy_gs)*(10**3), np.std(auc_greedy_gs_jac)*(10**3)]
    for i in range(5):
        mean_final_error.append(np.mean(result_residual[i][-1])*(10**3))
        std_final_error.append(np.std(result_residual[i][-1])*(10**3))
        mean_auc_error.append(np.mean(auc_errors[i])*(10**3))
        std_auc_error.append(np.std(auc_errors[i])*(10**3))

    df_residual[f"Mean_FinalResidual_{dim}d_{equation}"] = mean_final_error
    df_residual[f"Std_FinalResidual_{dim}d_{equation}"] = std_final_error

    df_residual[f"FinalResidual_{dim}d_{equation}"] = df_residual[f"Mean_FinalResidual_{dim}d_{equation}"].round(3).astype(str).str.cat(df_residual[f"Std_FinalResidual_{dim}d_{equation}"].round(3).astype(str), sep = " (") + ")"
    
    df_residual[f"Mean_AUC_Residual_{dim}d_{equation}"] = mean_auc_error
    df_residual[f"Std_AUC_Residual_{dim}d_{equation}"] = std_auc_error
    df_residual[f"AUC_Residual_{dim}d_{equation}"] = df_residual[f"Mean_AUC_Residual_{dim}d_{equation}"].round(3).astype(str).str.cat(df_residual[f"Std_AUC_Residual_{dim}d_{equation}"].round(3).astype(str), sep = " (") + ")"
    df_residual.set_index("Methods", inplace = True)
    df_residual.to_csv(f"results/increasingK_2_separate_{ml_model_type}_{ml_model_name}_residual_comparison_{model_type}router_{model_name}.csv")
    print("RESIDUAL COMPARISON")
    print(df_residual.to_latex(columns = ["FinalResidual_1d_Poisson", "AUC_Residual_1d_Poisson", "FinalResidual_2d_Poisson", "AUC_Residual_2d_Poisson", "FinalResidual_1d_Helmholtz", "AUC_Residual_1d_Helmholtz", "FinalResidual_2d_Helmholtz", "AUC_Residual_2d_Helmholtz"]))



    modeoneauc_greedy = np.trapezoid(mode_one_greedy, axis=0)
    # modeoneauc_greedy_gs = np.trapezoid(mode_one_greedy_gs, axis=0)
    # modeoneauc_greedy_gs_jac = np.trapezoid(mode_one_greedy_gs_jac, axis=0)
    auc_modeoneerrors = []
    for i in range(5):
        auc_modeoneerrors.append(np.trapezoid(result_mode_one[i], axis=0))

    modefiveauc_greedy = np.trapezoid(mode_five_greedy, axis=0)
    # modefiveauc_greedy_gs = np.trapezoid(mode_five_greedy_gs, axis=0)
    # modefiveauc_greedy_gs_jac = np.trapezoid(mode_five_greedy_gs_jac, axis=0)
    auc_modefiveerrors = []
    for i in range(5):
        auc_modefiveerrors.append(np.trapezoid(result_mode_five[i], axis=0))

    modetenauc_greedy = np.trapezoid(mode_ten_greedy, axis=0)
    # modetenauc_greedy_gs = np.trapezoid(mode_ten_greedy_gs, axis=0)
    # modetenauc_greedy_gs_jac = np.trapezoid(mode_ten_greedy_gs_jac, axis=0)
    auc_modetenerrors = []
    for i in range(5):
        auc_modetenerrors.append(np.trapezoid(result_mode_ten[i], axis=0))


    if os.path.exists(f"results/increasingK_2_separate_{ml_model_type}_{ml_model_name}_modes_comparison_{model_type}router_{model_name}.csv"):
        df_mode_error = pd.read_csv(f"results/increasingK_2_separate_{ml_model_type}_{ml_model_name}_modes_comparison_{model_type}router_{model_name}.csv")
    else:
        df_mode_error = pd.DataFrame({"Methods": ["1", "2", "3", "4", "5"]})# ["Greedy-Jacobi", "Greedy-GS", "2", "3", "4", "5"]})
        df_mode_error["Final_Mode1_1d_Poisson"] = ""
        df_mode_error["AUC_Mode1_1d_Poisson"] = ""
        df_mode_error["Final_Mode5_1d_Poisson"] = ""
        df_mode_error["AUC_Mode5_1d_Poisson"] = ""
        df_mode_error["Final_Mode10_1d_Poisson"] = ""
        df_mode_error["AUC_Mode10_1d_Poisson"] = ""

        df_mode_error["Final_Mode1_2d_Poisson"] = ""
        df_mode_error["AUC_Mode1_2d_Poisson"] = ""
        df_mode_error["Final_Mode5_2d_Poisson"] = ""
        df_mode_error["AUC_Mode5_2d_Poisson"] = ""
        df_mode_error["Final_Mode10_2d_Poisson"] = ""
        df_mode_error["AUC_Mode10_2d_Poisson"] = ""

        df_mode_error["Final_Mode1_1d_Helmholtz"] = ""
        df_mode_error["AUC_Mode1_1d_Helmholtz"] = ""
        df_mode_error["Final_Mode5_1d_Helmholtz"] = ""
        df_mode_error["AUC_Mode5_1d_Helmholtz"] = ""
        df_mode_error["Final_Mode10_1d_Helmholtz"] = ""
        df_mode_error["AUC_Mode10_1d_Helmholtz"] = ""

        df_mode_error["Final_Mode1_2d_Helmholtz"] = ""
        df_mode_error["AUC_Mode1_2d_Helmholtz"] = ""
        df_mode_error["Final_Mode5_2d_Helmholtz"] = ""
        df_mode_error["AUC_Mode5_2d_Helmholtz"] = ""
        df_mode_error["Final_Mode10_2d_Helmholtz"] = ""
        df_mode_error["AUC_Mode10_2d_Helmholtz"] = ""
    
    mean_final_mode_one = []# [np.mean(mode_one_greedy[-1])*(10**3), np.mean(mode_one_greedy_gs[-1])*(10**3), np.mean(mode_one_greedy_gs_jac[-1])*(10**3)]
    std_final_mode_one = []# [np.std(mode_one_greedy[-1])*(10**3), np.std(mode_one_greedy_gs[-1])*(10**3), np.std(mode_one_greedy_gs_jac[-1])*(10**3)]
    mean_auc_mode_one = []# [np.mean(modeoneauc_greedy)*(10**3), np.mean(modeoneauc_greedy_gs)*(10**3), np.mean(modeoneauc_greedy_gs_jac)*(10**3)]
    std_auc_mode_one = []# [np.std(modeoneauc_greedy)*(10**3), np.std(modeoneauc_greedy_gs)*(10**3), np.std(modeoneauc_greedy_gs_jac)*(10**3)]

    mean_final_mode_five = []#[np.mean(mode_five_greedy[-1])*(10**3), np.mean(mode_five_greedy_gs[-1])*(10**3), np.mean(mode_five_greedy_gs_jac[-1])*(10**3)]
    std_final_mode_five = []# [np.std(mode_five_greedy[-1])*(10**3), np.std(mode_five_greedy_gs[-1])*(10**3), np.std(mode_five_greedy_gs_jac[-1])*(10**3)]
    mean_auc_mode_five = []# [np.mean(modefiveauc_greedy)*(10**3), np.mean(modefiveauc_greedy_gs)*(10**3), np.mean(modefiveauc_greedy_gs_jac)*(10**3)]
    std_auc_mode_five = [] # [np.std(modefiveauc_greedy)*(10**3), np.std(modefiveauc_greedy_gs)*(10**3), np.std(modefiveauc_greedy_gs_jac)*(10**3)]

    mean_final_mode_ten = [] # [np.mean(mode_ten_greedy[-1])*(10**3), np.mean(mode_ten_greedy_gs[-1])*(10**3), np.mean(mode_ten_greedy_gs_jac[-1])*(10**3)]
    std_final_mode_ten = [] # [np.std(mode_ten_greedy[-1])*(10**3), np.std(mode_ten_greedy_gs[-1])*(10**3), np.std(mode_ten_greedy_gs_jac[-1])*(10**3)]
    mean_auc_mode_ten = [] # [np.mean(modetenauc_greedy)*(10**3), np.mean(modetenauc_greedy_gs)*(10**3), np.mean(modetenauc_greedy_gs_jac)*(10**3)]
    std_auc_mode_ten = [] # [np.std(modetenauc_greedy)*(10**3), np.std(modetenauc_greedy_gs)*(10**3), np.std(modetenauc_greedy_gs_jac)*(10**3)]

    for i in range(5):
        mean_final_mode_one.append(np.mean(result_mode_one[i][-1])*(10**3))
        std_final_mode_one.append(np.std(result_mode_one[i][-1])*(10**3))
        mean_auc_mode_one.append(np.mean(auc_modeoneerrors[i])*(10**3))
        std_auc_mode_one.append(np.std(auc_modeoneerrors[i])*(10**3))

        mean_final_mode_five.append(np.mean(result_mode_five[i][-1])*(10**3))
        std_final_mode_five.append(np.std(result_mode_five[i][-1])*(10**3))
        mean_auc_mode_five.append(np.mean(auc_modefiveerrors[i])*(10**3))
        std_auc_mode_five.append(np.std(auc_modefiveerrors[i])*(10**3))

        mean_final_mode_ten.append(np.mean(result_mode_ten[i][-1])*(10**3))
        std_final_mode_ten.append(np.std(result_mode_ten[i][-1])*(10**3))
        mean_auc_mode_ten.append(np.mean(auc_modetenerrors[i])*(10**3))
        std_auc_mode_ten.append(np.std(auc_modetenerrors[i])*(10**3))

    df_mode_error[f"Mean_Final_Mode1_{dim}d_{equation}"] = mean_final_mode_one
    df_mode_error[f"Std_Final_Mode1_{dim}d_{equation}"] = std_final_mode_one
    df_mode_error[f"Final_Mode1_{dim}d_{equation}"] = df_mode_error[f"Mean_Final_Mode1_{dim}d_{equation}"].round(3).astype(str).str.cat(df_mode_error[f"Std_Final_Mode1_{dim}d_{equation}"].round(3).astype(str), sep = " (") + ")"

    df_mode_error[f"Mean_AUC_Mode1_{dim}d_{equation}"] = mean_auc_mode_one
    df_mode_error[f"Std_AUC_Mode1_{dim}d_{equation}"] = std_auc_mode_one
    df_mode_error[f"AUC_Mode1_{dim}d_{equation}"] = df_mode_error[f"Mean_AUC_Mode1_{dim}d_{equation}"].round(3).astype(str).str.cat(df_mode_error[f"Std_AUC_Mode1_{dim}d_{equation}"].round(3).astype(str), sep = " (") + ")"

    df_mode_error[f"Mean_Final_Mode5_{dim}d_{equation}"] = mean_final_mode_five
    df_mode_error[f"Std_Final_Mode5_{dim}d_{equation}"] = std_final_mode_five
    df_mode_error[f"Final_Mode5_{dim}d_{equation}"] = df_mode_error[f"Mean_Final_Mode5_{dim}d_{equation}"].round(3).astype(str).str.cat(df_mode_error[f"Std_Final_Mode5_{dim}d_{equation}"].round(3).astype(str), sep = " (") + ")"

    df_mode_error[f"Mean_AUC_Mode5_{dim}d_{equation}"] = mean_auc_mode_five
    df_mode_error[f"Std_AUC_Mode5_{dim}d_{equation}"] = std_auc_mode_five
    df_mode_error[f"AUC_Mode5_{dim}d_{equation}"] = df_mode_error[f"Mean_AUC_Mode5_{dim}d_{equation}"].round(3).astype(str).str.cat(df_mode_error[f"Std_AUC_Mode5_{dim}d_{equation}"].round(3).astype(str), sep = " (") + ")"

    df_mode_error[f"Mean_Final_Mode10_{dim}d_{equation}"] = mean_final_mode_ten
    df_mode_error[f"Std_Final_Mode10_{dim}d_{equation}"] = std_final_mode_ten
    df_mode_error[f"Final_Mode10_{dim}d_{equation}"] = df_mode_error[f"Mean_Final_Mode10_{dim}d_{equation}"].round(3).astype(str).str.cat(df_mode_error[f"Std_Final_Mode10_{dim}d_{equation}"].round(3).astype(str), sep = " (") + ")"

    df_mode_error[f"Mean_AUC_Mode10_{dim}d_{equation}"] = mean_auc_mode_ten
    df_mode_error[f"Std_AUC_Mode10_{dim}d_{equation}"] = std_auc_mode_ten
    df_mode_error[f"AUC_Mode10_{dim}d_{equation}"] = df_mode_error[f"Mean_AUC_Mode10_{dim}d_{equation}"].round(3).astype(str).str.cat(df_mode_error[f"Std_AUC_Mode10_{dim}d_{equation}"].round(3).astype(str), sep = " (") + ")"
    df_mode_error.set_index("Methods", inplace = True)

    df_mode_error.to_csv(f"results/increasingK_2_separate_{ml_model_type}_{ml_model_name}_modes_comparison_{model_type}router_{model_name}.csv")
    print("Poisson 1d Mode COMPARISON")
    print(df_mode_error.to_latex(columns = ["Final_Mode1_1d_Poisson", "AUC_Mode1_1d_Poisson", "Final_Mode5_1d_Poisson", "AUC_Mode5_1d_Poisson", "Final_Mode10_1d_Poisson", "AUC_Mode10_1d_Poisson"]))


    print("Poisson 2d mode error COMPARISON")
    print(df_mode_error.to_latex(columns = ["Final_Mode1_2d_Poisson", "AUC_Mode1_2d_Poisson", "Final_Mode5_2d_Poisson", "AUC_Mode5_2d_Poisson", "Final_Mode10_2d_Poisson", "AUC_Mode10_2d_Poisson"]))

    print("Helmholtz 1d mode error COMPARISON")
    print(df_mode_error.to_latex(columns = ["Final_Mode1_1d_Helmholtz", "AUC_Mode1_1d_Helmholtz", "Final_Mode5_1d_Helmholtz", "AUC_Mode5_1d_Helmholtz", "Final_Mode10_1d_Helmholtz", "AUC_Mode10_1d_Helmholtz"]))

    print("Helmholtz 2d mode error COMPARISON")
    print(df_mode_error.to_latex(columns = ["Final_Mode1_2d_Helmholtz", "AUC_Mode1_2d_Helmholtz", "Final_Mode5_2d_Helmholtz", "AUC_Mode5_2d_Helmholtz", "Final_Mode10_2d_Helmholtz", "AUC_Mode10_2d_Helmholtz"]))



if __name__ == "__main__":
    print("Parsing arguments...")
    args, unknown = parser.parse_known_args()
    model_type = args.model
    ml_model_type = args.ml_model
    n_test = args.n_test
    ckp_dir = args.ckp_dir
    model_name = args.model_name
    ml_model_name = args.ml_model_name
    data_dir = args.data_dir
    extra = args.extra

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # run(model_type, ml_model_type, n_test, 2, "Periodic", "Helmholtz", ckp_dir, model_name, ml_model_name, data_dir, 1, device)
    run_2(model_type, ml_model_type, n_test, 1, "Periodic", "Poisson", ckp_dir, model_name, ml_model_name, data_dir, 1, device)
    # run(model_type, ml_model_type, n_test, 1, "Periodic", "Poisson", ckp_dir, model_name, ml_model_name, data_dir, 1, device)
    # run(model_type, ml_model_type, n_test, 2, "Periodic", "Poisson", ckp_dir, model_name, ml_model_name, data_dir, 1, device)
    exit()
    
