# import optuna
# import pandas as pd
# import torch
# import torch.nn.functional as F
# from utils.new_loss import New_Loss
# from utils.utils_model import evaluate
# # from polarization_random_search import 

# # Load your CSV with precomputed random search data
# df = pd.read_csv("polarization_random_search - Sheet1.csv")

# # If needed, rename columns to standard names used in the trial
# df.rename(columns={
#     'em_dim': 'em_dim',
#     'layers': 'layers',
#     'mul': 'mul',
#     'lmax': 'lmax',
#     'r_max': 'r_max',
#     'loss': 'loss'  # Ensure your loss column is named 'loss'
# }, inplace=True)

# # # Define the custom loss function (applied to loss values if needed)
# # def adjusted_loss(loss_val):
# #     input = torch.tensor([loss_val], dtype=torch.float32)
# #     target = torch.tensor([0.0], dtype=torch.float32)
# #     return F.mse_loss(10**input - 1, 10**target - 1, reduction='mean').item()

# # # Loss function to find new loss average
# # def new_loss_avg(losses):
# #     loss_fn = torch.nn.MSELoss()
# #     loss_fn_mae = torch.nn.L1Loss()
# #     new_loss = New_Loss()
# #     new_loss_avg = evaluate(model, dataloader_valid, new_loss, loss_fn_mae, device)

# # Create a lookup objective function that uses precomputed results
# def objective(trial):
#     em_dim = trial.suggest_categorical("em_dim", [16, 32, 64, 96, 128])
#     layers = trial.suggest_categorical("layers", [1, 2, 3, 4])
#     mul = trial.suggest_categorical("mul", [8, 16, 32, 64])
#     lmax = trial.suggest_categorical("lmax", [1, 2, 3, 4])
#     r_max = trial.suggest_float("r_max", 2, 6)

#     # Find matching rows in the DataFrame
#     matches = df[
#         (df["em_dim"] == em_dim) &
#         (df["layers"] == layers) &
#         (df["mul"] == mul) &
#         (df["lmax"] == lmax) &
#         (df["r_max"].round(4) == round(r_max, 4))  # Round float to avoid precision mismatch
#     ]

#     if matches.empty:
#         # Penalize unknown combinations
#         return float("inf")

#     raw_loss = matches.iloc[0]["New Loss Avg"]  # Use first match
#     return adjusted_loss(raw_loss)      # Apply your MSE-based loss transformation

# # Create the Optuna study
# # sampler = optuna.samplers.TPESampler(n_startup_trials=10, acquisition_function="EI")
# sampler = optuna.samplers.TPESampler(n_startup_trials=10)

# study = optuna.create_study(direction="minimize", sampler=sampler)

# # Run optimization
# study.optimize(objective, n_trials=50)

# # Show the best parameters and the associated transformed loss
# print("Best hyperparameters found:")
# print(study.best_params)
# print("Adjusted (transformed) loss:")
# print(study.best_value)


