# Set parameters that define an experiment
# Think of the columns of the arrays defining an experiment
# I.e., define an experiment along the same array index
n_agents=(10)
n_actions=(5)
threshold=(2) 
feature_histories=(1)

# How many times to run each experiment
n_runs=(3)

# Loop through experiments
for i in ${!n_runs[@]}; do
  # Create folder for experiment (if it doesn't exit)
  experiment_folder="./data/""${n_agents[i]}""agents_""${n_actions[i]}""actions_""${threshold[i]}""threshold_"
  experiment_folder="$experiment_folder""${feature_histories[i]}""history"

  # Create folder for individual run, could change to data file
  run_number=1
  full_dir="$experiment_folder""/run""$run_number""/"
  while [ -d "$full_dir" ]
  do
    let run_number+=1 
    full_dir="$experiment_folder""/run""$run_number""/"
  done

  # Loop through individual runs
  for ((j = 1 ; j <= ${n_runs[$i]} ; j++)); do
    # Create folder for run
    mkdir -p ./$full_dir

    # Run
    python argparse_agent.py ${n_agents[$i]} ${n_actions[$i]} ${threshold[$i]} ${feature_histories[$i]} $full_dir

    # Update new directory
    let run_number+=1
    full_dir="$experiment_folder""/run""$run_number""/"
  done 
done