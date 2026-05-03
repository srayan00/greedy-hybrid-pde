# A Greedy PDE Router for Blending Neural Operators and Classical Methods
  
## Install Dependencies
Run the following command to install all required dependencies
`conda env create -f environment.yml`

## Training DeepONet
See Table 3 in Appendix D for the exact hyperparameters used in our DeepONet and port them over to args/deeponet_args.json. Run the following commands:
`conda activate greedy`
`python train_ml_solver.py --model_name ML_MODEL_NAME --equation  [Poisson/ConvDiff]`

For example, 
`python train_ml_solver.py --model_name ml_example --equation  Poisson`


## Training a Greedy Router

Run the following command:
`python train_router.py --ml_model_name ML_MODEL_NAME --model_name MODEL_NAME --equation [Poisson/ConvDiff] --numerical_solvers LIST_OF_SOLVERS`
where `LIST_OF_SOLVERS` is a comma-separated list of solvers in the solver ensemble.

For example, 
`python train_router.py --ml_model_name ml_example --model_name example --equation Poisson --numerical_solvers jacobi_0.8,gs,sor_1.5,ssor`
where `jacobi_0.8` is a Weighted Jacobi solver with a relaxation parameter $\omega = 0.8$ and `gs` denotes Gauss-Seidel method, `sor_1.5` denotes a successive over-relaxation solver ($\omega = 1.5$), and `ssor` denotes a symmetric successive over-relaxation solver ($\omega = 1.0$)

## Running Experiments

### Comparing Greedy with HINTS experiment

Train routers for `equation` = `Poisson` and `ConvDiff` for the following list of solver ensembles `[jacobi, gs, ssor, jacobi_0.67, sor_1.5]` . There should be a total of $10$ routers ($2 \times 5$) 

After all these models are trained, run the command:
`python results.py --ml_model_name ML_MODEL_NAME --n_test 64 --model_name MODEL_NAME --equation [Poisson/ConvDiff] --numerical_solvers [jacobi/gs/ssor/jacobi_0.67/sor_1.5]`
for all 10 combinations. All the results (plots and tables) can be in the results folder

### Size of solver ensembles
Train routers for `equation` = `Poisson` and `Helmholtz` for the following list of solver ensembles:
* `jacobi,gs`
* `jacobi,gs,ssor`
* `jacobi,gs,ssor,jacobi_0.67`
* `jacobi,gs,ssor,jacobi_0.67,sor_1.5`


There should be a total of $8$ routers ($2 \times 4$) 

After all these models are trained, run the command:
`python multiple_solver_results.py --ml_model_name ML_MODEL_NAME --n_test 64 --model_name MODEL_NAME --equation [Poisson/ConvDiff] --numerical_solvers LIST_OF_SOLVERS`
for all $8$ combinations. All the results (plots and tables) can be found in the results folder


