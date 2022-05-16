import gym
from custom_env import threshold_env
from DQN import KerasDQN
import numpy as np
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("n_agents", type=int)
parser.add_argument("n_actions", type=int)
parser.add_argument("threshold", help="How many agents can successfully transmit at the same time", type=int)
parser.add_argument("feature_histories", help="Use information from the past x steps as features", type=int)
parser.add_argument("save_data_path", type=str)
args = parser.parse_args()

n_agents = args.n_agents
n_actions = args.n_actions
threshold = args. threshold
feature_histories = args.feature_histories
save_data_path = args.save_data_path

# --------------------> Parameters <--------------------
save = True # Save data to CSV
n_iterations = 1 # How many full simulations to run
# ------------------------------------------------------------

# --------------------- Create Env ---------------------
n_steps = 1e4 
transmit_and_sense = False
env = threshold_env(n_agents, threshold, n_steps, transmit_and_sense=transmit_and_sense)
# ------------------------------------------------------


def state_to_observations(state):
    """
    Input:
        - obs [list or np.array]: Concatenated list of all observations

    Returns:
        - list of lists of observations for each agent
    """
    n_obs_per_agent = len(state) // n_agents
    #print("n_obs_per_agent", n_obs_per_agent)
    agent_obs = [np.array(state[i * n_obs_per_agent: (i + 1) * n_obs_per_agent]).reshape(1, -1) for i in range(n_agents)]

    return agent_obs

# ---------------------- Training Loop --------------------
currIt = 0
while True:
  # --------------------- Create Agents ---------------------
  n_inputs = 4 * feature_histories 
  # DQN
  agents = [KerasDQN(n_inputs, n_actions,
                    hidden_layer_one_dims=128,
                    hidden_layer_two_dims=256,
                    batch_size=64,
                    epsilon_min=0.05) for _ in range(n_agents)]
  # ------------------------------------------------------

  stepIdx = 0
  rewards = []
  action_list = []
  states = []
  scores = [[] for _ in range(n_agents)] # is this the same as rewards?
  rewards = []

  state = env.reset() # If I refactor state, make this work
  state = [np.zeros(n_inputs).reshape(1, -1) for _ in range(n_agents)]
  next_state = [np.zeros(n_inputs).reshape(1, -1) for _ in range(n_agents)]
  print("starting next state", next_state)

  # For multi-step actions
  state_at_action = [np.zeros(n_inputs).reshape(1, -1) for _ in range(n_agents)]
  future_actions = [[] for _ in range(n_agents)]
  action_duration = [0 for _ in range(n_agents)]
  reward_over_actions = [[] for _ in range(n_agents)]
  actions = [0 for _ in range(n_agents)] # Action selected by the agent (could be multi-step)
  actions_to_take = [0 for _ in range(n_agents)] # do/don't transmit on this step. In {0, 1}
  
  while True:
    #print("\nStep", stepIdx) 
    # Get Actions ------------------------------
    for i in range(n_agents):
      # if buffer is 0 don't use RL, also don't save if no RL was used
      if state[i][0][-1] == 0:
        #actions.append(-1) # Original
        actions[i] = -1
        future_actions[i] = [-1]

      # If the action_duration is 0, get a new action,
      elif action_duration[i] == 0: # make sure this can't be negative
        # Get action, save state, set future actions, and action_duration
        agent_action = agents[i].choose_action(state[i])
        #print("agent", i, "action", agent_action)

        state_at_action[i] = state[i]

        if agent_action == 0:
          future_actions[i] = [0]
        elif agent_action == 1:
          future_actions[i] = [1]
        elif agent_action == 2:
          future_actions[i] = [0, 1]
        elif agent_action == 3:
          future_actions[i] = [0, 0, 1]
        elif agent_action == 4:
          future_actions[i] = [0, 0, 0, 1]
        elif agent_action == 5:
          future_actions[i] = [0, 0, 0, 0, 1]
        elif agent_action == 6:
          future_actions[i] = [0, 0, 0, 0, 0, 1]
        elif agent_action == 7:
          future_actions[i] = [0, 0, 0, 0, 0, 0, 1]
        elif agent_action == 8:
          future_actions[i] = [0, 0, 0, 0, 0, 0, 0, 1]
        elif agent_action == 9:
          future_actions[i] = [0, 0, 0, 0, 0, 0, 0, 0, 1]
        elif agent_action == 10:
          future_actions[i] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        else:
          raise ValueError

        # Update actions if a new decision is made
        actions[i] = agent_action

      action_duration[i] = len(future_actions[i])

      # Set action to take by popping future action
      actions_to_take[i] = future_actions[i].pop(0)
    # -------------------------------------------------------------------

    # Take an environment step
    new_state_info, reward, done, info = env.step(actions_to_take)
    #print("next state before", new_state_info)

    """
    # Feature history stuff (testing)
    for i in range(n_agents):
      # Shift elements to the left
      for j in range(len(next_state[i][0]) - len(new_state_info[i])):
        next_state[i][0][j] = next_state[i][0][j + len(new_state_info[i])]
      
      # Add new state information to the end of the list
      for k in range(len(new_state_info[i])):
        next_state[i][0][-1 * len(new_state_info[i]) + k] = new_state_info[i][k]
    """
    next_state = state_to_observations(new_state_info)
    #print("next state after", next_state, "\n")
    """
    print("state", state)
    print("actions to take", actions_to_take)
    print("next_state", next_state)
    print("reward", reward)
    print("done", done)
    print("info", info)
    """

    # Decrement all action durations
    action_duration = [duration - 1 for duration in action_duration]

    # Remember reward and transitions
    for i in range(n_agents):
      agent_action = actions[i]
      #print("action", agent_action)

      # Check if RL was not used
      if agent_action == -1: # RL agent not invoked. Do not save transition to memory
        continue

      # Add reward to reward_over_actions
      agent_reward = reward[i] # For now, reward is the same for all agents
      reward_over_actions[i].append(agent_reward)

      # Save transitions only when action_duration == 0
      if action_duration[i] == 0:
        agent_state = state_at_action[i]
        agent_next_state = next_state[i]

        # Average reward
        agent_average_reward_over_action = float(np.mean(reward_over_actions[i]))
        # Save transition with the state at the time of the action decision and
        # the average reward over the course of the action
        agents[i].remember(agent_state, agent_action, agent_average_reward_over_action,
                  agent_next_state, done)

        agents[i].learn() # Could be moved outside of the "if" block

        # Clear reward_over_actions
        reward_over_actions = [[] for _ in range(n_agents)]


    for i in range(n_agents):
      scores[i].append(reward[i])


    # print("reward", reward)
    # print("acitons", actions)
    # print("states", state)
    rewards.append(reward.copy())
    action_list.append(actions.copy())
    states.append(state)
    
    # print(len(rewards), reward)
    # print(len(action_list), action_list)
    # print(len(states), states)
    # print()
    state = next_state

    stepIdx += 1
    if stepIdx % 100 == 0:
      print("Step: ", stepIdx)
      for i in range(n_agents):
        print("mean (last 50)", np.mean(scores[i][-50:]))
        if i == (n_agents - 1):
          print()

    if done:
      # Record data in CSV
      if save == True:
        data = [list(reward) + list(action) + list(np.array(state).flatten()) for reward, action, state in zip(rewards, action_list, states)]
        with open(save_data_path + "data" + str(currIt) + ".csv", "w", newline="") as f:
          writer = csv.writer(f)
          writer.writerows(data)
      break

  currIt += 1
  if currIt == n_iterations:
    break
