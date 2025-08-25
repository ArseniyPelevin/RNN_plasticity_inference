# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
# ---

# +

config = {
    "num_inputs": 100,  # x
    "num_hidden": 1000,  # y
    "num_outputs": 1,  # m
    "num_exp": 50,  # Number of experiments/trajectories
    "num_steps": 50,  # Number of steps per experiment
    "num_epochs": 250,
    "generation_plasticity": "1X1Y1W0R0-1X0Y2W1R0",
}

# Generate model activity

# Initialize parameters for training

# Training

for epoch in range(config["num_epochs"]):
    # Simulate

    # Compute loss

    # Compute gradients

    # Update parameters

    # Print loss every 10 epochs
    if epoch % 10 == 0:
        loss = []
        print(f"Epoch {epoch}, Loss: {loss}")
