# A Greedy PDE Router for Blending Neural Operators and Classical Methods
  
## Install Dependencies
Run the following command to install all required dependencies
`conda env create -f environment.yml`

## Training DeepONet
See Table 3 in Appendix D for the exact hyperparameters used in our DeepONet and port them over to args/deeponet_args.json. Run the following commands:
`conda activate greedy`
`python train_ml_solver.py --model_name ML_MODEL_NAME --equation  [Poisson/Helmholtz] --dim  [1/2]`

For example, 
`python train_ml_solver.py --model_name ml_example --equation  Helmholtz --dim  2`


## Training a Greedy Router

Run the following command:
`python train_router.py --ml_model_name ML_MODEL_NAME --dim [1/2] --model_name MODEL_NAME --equation  [Poisson/Helmholtz] --numerical_solvers LIST_OF_SOLVERS`
where `LIST_OF_SOLVERS` is a comma-separated list of solvers in the solver ensemble.

For example, 
`python train_router.py --ml_model_name ml_example --dim 1 --model_name example --equation Poisson --numerical_solvers jacobi_0.8,gs,mg`
where `jacobi_0.8` is a Weighted Jacobi solver with a relaxation parameter $\omega = 0.8$, `gs` denotes Gauss-Seidel method, and `mg` denotes a multigrid solver (2 grid)

## Running Experiments

### Comparing Greedy with HINTS experiment

Train routers for `dim` = 1, 2 and `equation` = `Poisson` and `Helmholtz` for the following list of solver ensembles `[jacobi, gs, mg, gs,jacobi]` . There should be a total of $16$ routers ($2 \times 2 \times 4$) 

### Size of solver ensembles
Train routers for `dim` = 1 and `equation` = `Poisson` and `Helmholtz` for the following list of solver ensembles:
* `jacobi_0.5`
* `jacobi_0.5,jacobi_0.67`
* `jacobi_0.5,jacobi_0.67,jacobi_0.75`
* `jacobi_0.5,jacobi_0.67,jacobi_0.75,jacobi_0.8`
* `jacobi_0.5,jacobi_0.67,jacobi_0.75,jacobi_0.8,jacobi_1`


There should be a total of $10$ routers ($2 \times 5$) 

After all these models are trained, run the command:
`python experiments.py --ml_model_name ML_MODEL_NAME --n_test 64 --model_name MODEL_NAME`
