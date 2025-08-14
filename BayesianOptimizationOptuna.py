import torch
import pandas as pd
import optuna
import os
from datetime import datetime
from polarization_random_search import PeriodicNetwork, train, evaluate, New_Loss, dataloader_train, dataloader_valid, n_train

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create Dataframe from csv file
# csv_path = "polarization_random_search - Sheet1.csv"
# if os.path.exists(csv_path):
#     df = pd.read_csv(csv_path)
# else:
#     df = pd.DataFrame(columns=["em_dim", "layers", "mul", "lmax", "r_max", "New Loss Avg"])

df = pd.read_csv("polarization_random_search - Sheet1.csv")

# Cache to avoid recomputation
def trial_already_ran(params, df):
    return not df[
        (df["em_dim"] == params["em_dim"]) &
        (df["layers"] == params["layers"]) &
        (df["mul"] == params["mul"]) &
        (df["lmax"] == params["lmax"]) &
        (df["r_max"].round(4) == round(params["r_max"], 4))
    ].empty

# Training + evaluation
def objective(trial):
    global df
    # Define search space
    em_dim = trial.suggest_int("em_dim", 16, 128)
    layers = trial.suggest_int("layers", 1, 4)
    mul = trial.suggest_int("mul", 8, 64)
    lmax = trial.suggest_int("lmax", 1, 4)
    r_max = trial.suggest_float("r_max", 2.0, 6.0)

    params = {
        "em_dim": em_dim,
        "layers": layers,
        "mul": mul,
        "lmax": lmax,
        "r_max": r_max
    }

    # Avoid re-evaluating duplicate trials
    if trial_already_ran(params, df):
        trial.set_user_attr("duplicate", True)
        existing_loss = df[
            (df["em_dim"] == em_dim) &
            (df["layers"] == layers) &
            (df["mul"] == mul) &
            (df["lmax"] == lmax) &
            (df["r_max"].round(4) == round(r_max, 4))
        ]["New Loss Avg"].values[0]
        return existing_loss

    # Build existing model from the original model
    model = PeriodicNetwork(
        in_dim=118,
        em_dim=em_dim,
        irreps_in=f"{em_dim}x0e",
        irreps_out="1x0e",
        irreps_node_attr=f"{em_dim}x0e",
        layers=layers,
        mul=mul,
        lmax=lmax,
        max_radius=r_max,
        num_neighbors=n_train.mean(),
        reduce_output=True
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.96)
    loss_fn = torch.nn.MSELoss()
    loss_fn_mae = torch.nn.L1Loss()
    new_loss = New_Loss()

    run_name = f"model_polarization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Train
    model.pool = True
    train(model, opt, dataloader_train, dataloader_valid, loss_fn, loss_fn_mae,
          run_name, max_iter=100, scheduler=scheduler, device=device)

    # Evaluate new_loss_avg
    new_loss_avg = evaluate(model, dataloader_valid, new_loss, loss_fn_mae, device)

    # Log results
    new_row = params.copy()
    new_row["New Loss Avg"] = new_loss_avg
    # global df
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    # df.to_csv(csv_path, index=False)

    return new_loss_avg

# Run optimization loop
study = optuna.create_study(direction="minimize") # Uses default TPE sampler
study.optimize(objective, n_trials=30)  

print("Best trial:")
print(study.best_trial)
