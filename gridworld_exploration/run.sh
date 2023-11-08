
python3 main.py --data maze --exo_noise two_maze --num_exo 1 --env_iteration 25000 --rows 11 --cols 11 --ncodes 120 --stochastic_start deterministic --ep_length 200 --no_restart true --max_kl_penalty 0.0005 --use_logger --folder ./results/goal_seek/ --heatmap_random false

cat results/test/maze/two_maze/rows_11_cols_11/DP-Goal/deterministic/no_restart_True/log.txt | ../dp/trace_to_tables | ../dp/state_metrics

python3 main.py --data maze --exo_noise two_maze --num_exo 1 --env_iteration 25000 --rows 11 --cols 11 --ncodes 120 --stochastic_start deterministic --ep_length 200 --no_restart true --max_kl_penalty 0.0005 --use_logger --folder ./results/random_rollout/ --heatmap_random true



python3 main.py --data maze --exo_noise two_maze --num_exo 8 --env_iteration 50000 --rows 11 --cols 11 --ncodes 120 --stochastic_start deterministic --ep_length 200 --no_restart true --max_kl_penalty 0.0005 --use_logger


#Maybe 6000 iterations
python3 main.py --data maze --exo_noise two_maze --num_exo 8 --env_iteration 10000 --rows 11 --cols 11 --ncodes 620 --stochastic_start deterministic --ep_length 400 --no_restart true --max_kl_penalty 0.0001 --use_logger --folder ./results_trial_multi_maze

#Takes maybe 3000 iterations
python3 main.py --data vis-maze --exo_noise multi_maze --num_exo 8 --env_iteration 10000 --rows 5 --cols 5 --ncodes 620 --stochastic_start deterministic --ep_length 400 --no_restart true --max_kl_penalty 0.00001 --use_logger --folder ./results_trial_multi_maze




# python3 main.py --data maze --exo_noise two_maze --num_exo 8 --env_iteration 10000 --rows 11 --cols 11 --ncodes 620 --stochastic_start deterministic --ep_length 400 --no_restart true --max_kl_penalty 0.0001 --use_logger --folder ./results_trial_multi_maze

# #Takes maybe 3000 iterations
# python3 main.py --data vis-maze --exo_noise multi_maze --num_exo 8 --env_iteration 10000 --rows 5 --cols 5 --ncodes 64 --stochastic_start deterministic --ep_length 400 --no_restart true --use_logger --folder ./results_trial_multi_maze




### testing and debugging for 9-maze spiral-mazes

python3 main.py --data vis-maze --exo_noise multi_maze --num_exo 1 --env_iteration 10000 --rows 4 --cols 4 --ncodes 36 --stochastic_start deterministic --ep_length 400 --no_restart true --use_logger --folder ./results_trial_multi_maze

python3 main.py --data vis-maze --exo_noise multi_maze --num_exo 8 --env_iteration 10000 --rows 4 --cols 4 --ncodes 36 --stochastic_start deterministic --ep_length 400 --no_restart true --use_logger --folder ./results_trial_multi_maze --walls spiral --obs_type pixels

## for spiralmazes

python3 main.py --data vis-maze --exo_noise multi_maze --num_exo 1 --env_iteration 10000 --rows 4 --cols 4 --ncodes 620 --stochastic_start deterministic --ep_length 400 --no_restart true --use_logger --folder ./results_trial_multi_maze

cat results_trial_multi_maze/test/spiralworld/multi_maze/rows_4_cols_4/DP-Goal/deterministic/no_restart_True/log.txt | ../dp/trace_to_tables | ../dp/state_metrics



python3 main.py --data vis-maze --exo_noise multi_maze --num_exo 8 --env_iteration 10000 --rows 4 --cols 4 --ncodes 620 --stochastic_start deterministic --ep_length 400 --no_restart true --use_logger --folder ./results_trial_multi_maze

cat results_trial_multi_maze/test/spiralworld/multi_maze/rows_4_cols_4/DP-Goal/deterministic/no_restart_True/log.txt | ../dp/trace_to_tables | ../dp/state_metrics



python3 main.py --data vis-maze --exo_noise multi_maze --num_exo 1 --env_iteration 10000 --rows 6 --cols 6 --ncodes 620 --stochastic_start deterministic --ep_length 400 --no_restart true --use_logger --folder ./results_trial_multi_maze

cat results_trial_multi_maze/test/spiralworld/multi_maze/rows_6_cols_6/DP-Goal/deterministic/no_restart_True/log.txt | ../dp/trace_to_tables | ../dp/state_metrics



python3 main.py --data vis-maze --exo_noise multi_maze --num_exo 8 --env_iteration 10000 --rows 6 --cols 6 --ncodes 620 --stochastic_start deterministic --ep_length 400 --no_restart true --use_logger --folder ./results_trial_multi_maze

cat results_trial_multi_maze/test/spiralworld/multi_maze/rows_6_cols_6/DP-Goal/deterministic/no_restart_True/log.txt | ../dp/trace_to_tables | ../dp/state_metrics


### pixel based env
python3 main.py --data vis-maze --exo_noise multi_maze --num_exo 1 --env_iteration 10000 --rows 4 --cols 4 --ncodes 620 --stochastic_start deterministic --ep_length 400 --no_restart true --use_logger --folder ./results_trial_multi_maze --obs_type pixels


python3 main.py --data vis-maze --exo_noise multi_maze --num_exo 8 --env_iteration 10000 --rows 4 --cols 4 --ncodes 620 --stochastic_start deterministic --ep_length 400 --no_restart true --use_logger --folder ./results_trial_multi_maze --obs_type pixels







## for abstract four rooms domain 
python3 main.py --data maze --exo_noise two_maze --num_exo 8 --env_iteration 10000 --rows 8 --cols 8 --ncodes 620 --stochastic_start deterministic --ep_length 400 --no_restart true --use_logger --folder ./results_trial_multi_maze

cat results_trial_multi_maze/test/maze/two_maze/rows_8_cols_8/DP-Goal/deterministic/no_restart_True//log.txt | ../dp/trace_to_tables | ../dp/state_metrics

python3 main.py --data maze --exo_noise two_maze --num_exo 8 --env_iteration 10000 --rows 4 --cols 4 --ncodes 620 --stochastic_start deterministic --ep_length 400 --no_restart true --use_logger --folder ./results_trial_multi_maze




## evaluating test run
cat results_trial_multi_maze/test/spiralworld/multi_maze/rows_4_cols_4/DP-Goal/deterministic/no_restart_True/log.txt | ../dp/trace_to_tables | ../dp/state_metrics


#### Evaluating results
cat results_multi_maze_high_dim/genik/results_deterministic/run_DP-Goal/spiralworld/multi_maze/rows_4_cols_4/DP-Goal/deterministic/no_restart_True/log.txt | ../dp/trace_to_tables | ../dp/state_metrics | tail -n 500

cat results_multi_maze_high_dim/genik/results_deterministic/run_DP-Goal/spiralworld/multi_maze/rows_4_cols_4/DP-Goal/deterministic/no_restart_True/log.txt | ../dp/trace_to_tables | ../dp/state_metrics | head -n 500

cat results_multi_maze_high_dim/genik/results_deterministic/run_DP-Goal/spiralworld/multi_maze/rows_5_cols_5/DP-Goal/deterministic/no_restart_True/log.txt | ../dp/trace_to_tables | ../dp/state_metrics

cat results_multi_maze_high_dim/genik/results_deterministic/run_DP-Goal/spiralworld/multi_maze/rows_6_cols_6/DP-Goal/deterministic/no_restart_True/log.txt | ../dp/trace_to_tables | ../dp/state_metrics

cat results_multi_maze_high_dim/genik/results_deterministic/run_DP-Goal/spiralworld/multi_maze/rows_8_cols_8/DP-Goal/deterministic/no_restart_True/log.txt | ../dp/trace_to_tables | ../dp/state_metrics



cat results_multi_maze_pixels/genik/results_deterministic/run_DP-Goal/spiralworld/multi_maze/rows_4_cols_4/DP-Goal/deterministic/no_restart_True/log.txt | ../dp/trace_to_tables | ../dp/state_metrics

cat results_multi_maze_pixels/genik/results_deterministic/run_DP-Goal/spiralworld/multi_maze/rows_5_cols_5/DP-Goal/deterministic/no_restart_True/log.txt | ../dp/trace_to_tables | ../dp/state_metrics

cat results_multi_maze_pixels/genik/results_deterministic/run_DP-Goal/spiralworld/multi_maze/rows_6_cols_6/DP-Goal/deterministic/no_restart_True/log.txt | ../dp/trace_to_tables | ../dp/state_metrics

cat results_multi_maze_pixels/genik/results_deterministic/run_DP-Goal/spiralworld/multi_maze/rows_8_cols_8/DP-Goal/deterministic/no_restart_True/log.txt | ../dp/trace_to_tables | ../dp/state_metrics

















### testing for multi-maze for pixel based spiralmazes

python main.py --data vis-maze --exo_noise multi_maze --rows 8 --cols 8 --ncodes 120 --stochastic_start deterministic --ep_length 400 --no_restart true --use_logger --folder ./results_trial_multi_maze

cat results_trial_multi_maze/test/spiralworld/multi_maze/rows_8_cols_8/DP-Goal/deterministic/no_restart_True/log.txt | ../dp/trace_to_tables | ../dp/state_metrics


python main.py --data vis-maze --exo_noise multi_maze --rows 6 --cols 6 --ncodes 120 --stochastic_start deterministic --ep_length 400 --no_restart true --use_logger --folder ./results_trial_multi_maze

cat results_trial_multi_maze/test/spiralworld/multi_maze/rows_6_cols_6/DP-Goal/deterministic/no_restart_True/log.txt | ../dp/trace_to_tables | ../dp/state_metrics


## pixel multi_maze
python main.py --data vis-maze --exo_noise multi_maze --obs_type pixels --rows 8 --cols 8 --ncodes 100 --stochastic_start deterministic --ep_length 400 --no_restart true --use_logger --folder ./results_trial_multi_maze

cat results_trial_multi_maze/test/spiralworld/multi_maze/rows_8_cols_8/DP-Goal/deterministic/no_restart_True/log.txt | ../dp/trace_to_tables | ../dp/state_metrics


## 2x2 Multi-Maze 
python main.py --data vis-maze --exo_noise multi_maze --num_exo 3 --num_rand_initial 1 --rows 6 --cols 6 --ncodes 120 --stochastic_start deterministic --ep_length 400 --no_restart true --use_logger --folder ./results_trial_multi_maze




