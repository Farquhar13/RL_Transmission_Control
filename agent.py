from CSMA_agent import CsmaAgent
import gym
from custom_env import threshold_env
from DQN import KerasDQN
import numpy as np
import csv

# --------------------> Parameters <--------------------
save = True # Save data to CSV
save_data_path = "./data/"
n_iterations = 3 # How many full simulations to run
feature_histories = 1
# ------------------------------------------------------------

# --------------------- Create Env ---------------------
n_agents = 4 
threshold = 1 
n_steps = 1e4
transmit_and_sense = False
# With buffer intervals
buffer_intervals = [2, 5,  8, 10] * 4
env = threshold_env(n_agents, threshold, n_steps, 
                    transmit_and_sense=transmit_and_sense,
                    buffer_intervals=buffer_intervals)
"""
env = threshold_env(n_agents, threshold, n_steps, 
                    transmit_and_sense=transmit_and_sense)
"""
# -----------------------------------------------------


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
  n_actions = 5 
  # DQN
  """
  agents = [KerasDQN(n_inputs, n_actions,
                    hidden_layer_one_dims=128,
                    hidden_layer_two_dims=256,
                    batch_size=64,
                    epsilon_min=0.05) for _ in range(n_agents)]
  """
  # CSMA Agents
  #agents = [CsmaAgent(wait_for_idle=True)  for _ in range(n_agents)]
  #agents = [CsmaAgent(wait_for_idle=True, back_off_strategy="fixed", p=n_actions)  for _ in range(n_agents)]
  #agents = [CsmaAgent(wait_for_idle=False)  for _ in range(n_agents)] # not used in paper
  agents = [CsmaAgent(wait_for_idle=False, back_off_strategy="fixed", p=n_actions)  for _ in range(n_agents)]
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

  # For multi-step actions
  state_at_action = [np.zeros(n_inputs).reshape(1, -1) for _ in range(n_agents)]
  future_actions = [[] for _ in range(n_agents)]
  action_duration = [0 for _ in range(n_agents)]
  reward_over_actions = [[] for _ in range(n_agents)]
  actions = [0 for _ in range(n_agents)] # Action selected by the agent (could be multi-step)
  actions_to_take = [0 for _ in range(n_agents)] # do/don't transmit on this step. In {0, 1}
  
  while True:
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

        """
        # Idea to try for 10 agents
        if agent_action == 0:
          future_actions[i] = [0]
        elif agent_action == 1:
          future_actions[i] = [1]
        elif agent_action == 2:
          future_actions[i] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        """

        # Update actions if a new decision is made
        actions[i] = agent_action

      action_duration[i] = len(future_actions[i])

      # Set action to take by popping future action
      actions_to_take[i] = future_actions[i].pop(0)
    # -------------------------------------------------------------------

    # Take an environment step
    new_state_info, reward, done, info = env.step(actions_to_take)
    next_state = state_to_observations(new_state_info)

    # Decrement all action durations
    action_duration = [duration - 1 for duration in action_duration]

    # Remember reward and transitions
    for i in range(n_agents):
      agent_action = actions[i]

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

        #print("actions", actions)

    for i in range(n_agents):
      scores[i].append(reward[i])

    rewards.append(reward.copy())
    action_list.append(actions.copy())
    states.append(state)

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
