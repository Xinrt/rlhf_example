#!/bin/bash

# Define the commands to run and the log files
declare -a commands=(
    "python3 run.py train_policy_with_original_rewards Hopper-v4 --n_envs 16 --million_timesteps 10 > hopper_original.log 2>&1"
    "python3 run.py train_policy_with_original_rewards BipedalWalker-v3 --n_envs 16 --million_timesteps 10 > bipedal_original.log 2>&1"
    "python3 run.py train_policy_with_original_rewards Swimmer-v4 --n_envs 16 --million_timesteps 10 > swimmer_original.log 2>&1"
    "python3 run.py train_policy_with_original_rewards HalfCheetah-v4 --n_envs 16 --million_timesteps 10 > halfcheetah_original.log 2>&1"
    "python3 run.py train_policy_with_original_rewards Ant-v4 --n_envs 16 --million_timesteps 10 > ant_original.log 2>&1"
    "python3 run.py train_policy_with_original_rewards Reacher-v4 --n_envs 16 --million_timesteps 10 > reacher_original.log 2>&1"
    "python3 run.py train_policy_with_original_rewards Pendulum-v1 --n_envs 16 --million_timesteps 10 > pendulum_original.log 2>&1"
    "python3 run.py train_policy_with_original_rewards InvertedDoublePendulum-v4 --n_envs 16 --million_timesteps 10 > inverted_double_pendulum_original.log 2>&1"
    "python3 run.py train_policy_with_preferences Hopper-v4 --synthetic_prefs --ent_coef 0.02 --million_timesteps 0.15 > hopper_prefs.log 2>&1"
    "python3 run.py train_policy_with_preferences BipedalWalker-v3 --synthetic_prefs --ent_coef 0.02 --million_timesteps 0.15 > bipedal_prefs.log 2>&1"
    "python3 run.py train_policy_with_preferences Swimmer-v4 --synthetic_prefs --ent_coef 0.02 --million_timesteps 0.15 > swimmer_prefs.log 2>&1"
    "python3 run.py train_policy_with_preferences HalfCheetah-v4 --synthetic_prefs --ent_coef 0.02 --million_timesteps 0.15 > halfcheetah_prefs.log 2>&1"
    "python3 run.py train_policy_with_preferences Ant-v4 --synthetic_prefs --ent_coef 0.02 --million_timesteps 0.15 > ant_prefs.log 2>&1"
    "python3 run.py train_policy_with_preferences Reacher-v4 --synthetic_prefs --ent_coef 0.02 --million_timesteps 0.15 > reacher_prefs.log 2>&1"
    "python3 run.py train_policy_with_preferences Pendulum-v1 --synthetic_prefs --ent_coef 0.02 --million_timesteps 0.15 > pendulum_prefs.log 2>&1"
    "python3 run.py train_policy_with_preferences InvertedDoublePendulum-v4 --synthetic_prefs --ent_coef 0.02 --million_timesteps 0.15 > inverted_double_pendulum_prefs.log 2>&1"
)

# Run all commands
for cmd in "${commands[@]}"; do
    echo "Running: $cmd"
    eval $cmd
done
