#!/bin/bash

# Define the commands to run and the log files
declare -a commands=(
    "python3 run.py train_policy_with_original_rewards Hopper-v4 --n_envs 16 --million_timesteps 10 | tee run_log/hopper_original.log"
    "python3 run.py train_policy_with_original_rewards BipedalWalker-v3 --n_envs 16 --million_timesteps 10 | tee run_log/bipedal_original.log"
    "python3 run.py train_policy_with_original_rewards Swimmer-v4 --n_envs 16 --million_timesteps 10 | tee run_log/swimmer_original.log"
    "python3 run.py train_policy_with_original_rewards HalfCheetah-v4 --n_envs 16 --million_timesteps 10 | tee run_log/halfcheetah_original.log"
    "python3 run.py train_policy_with_original_rewards Ant-v4 --n_envs 16 --million_timesteps 10 | tee run_log/ant_original.log"
    "python3 run.py train_policy_with_original_rewards Reacher-v4 --n_envs 16 --million_timesteps 10 | tee run_log/reacher_original.log"
    "python3 run.py train_policy_with_original_rewards Pendulum-v1 --n_envs 16 --million_timesteps 10 | tee run_log/pendulum_original.log"
    "python3 run.py train_policy_with_original_rewards InvertedDoublePendulum-v4 --n_envs 16 --million_timesteps 10 | tee run_log/inverted_double_pendulum_original.log"
    "python3 run.py train_policy_with_preferences Hopper-v4 --synthetic_prefs --ent_coef 0.02 --million_timesteps 0.15 | tee run_log/hopper_prefs.log"
    "python3 run.py train_policy_with_preferences BipedalWalker-v3 --synthetic_prefs --ent_coef 0.02 --million_timesteps 0.15 | tee run_log/bipedal_prefs.log"
    "python3 run.py train_policy_with_preferences Swimmer-v4 --synthetic_prefs --ent_coef 0.02 --million_timesteps 0.15 | tee run_log/swimmer_prefs.log"
    "python3 run.py train_policy_with_preferences HalfCheetah-v4 --synthetic_prefs --ent_coef 0.02 --million_timesteps 0.15 | tee run_log/halfcheetah_prefs.log"
    "python3 run.py train_policy_with_preferences Ant-v4 --synthetic_prefs --ent_coef 0.02 --million_timesteps 0.15 | tee run_log/ant_prefs.log"
    "python3 run.py train_policy_with_preferences Reacher-v4 --synthetic_prefs --ent_coef 0.02 --million_timesteps 0.15 | tee run_log/reacher_prefs.log"
    "python3 run.py train_policy_with_preferences Pendulum-v1 --synthetic_prefs --ent_coef 0.02 --million_timesteps 0.15 | tee run_log/pendulum_prefs.log"
    "python3 run.py train_policy_with_preferences InvertedDoublePendulum-v4 --synthetic_prefs --ent_coef 0.02 --million_timesteps 0.15 | tee run_log/inverted_double_pendulum_prefs.log"
)

# Run all commands
for cmd in "${commands[@]}"; do
    echo "Running: $cmd"
    eval $cmd
done
