from rl.rl_agent import RLAgent
from map_loader import load_map
from simulation.traffic_simulator import simulate_traffic
import random

G = load_map()
agent = RLAgent()

nodes = list(G.nodes())

EPISODES = 200

for ep in range(EPISODES):

    G_temp = simulate_traffic(G.copy())

    start = random.choice(nodes)
    goal = random.choice(nodes)

    state = start

    for step in range(50):

        neighbors = list(G_temp.neighbors(state))

        if not neighbors:
            break

        action = agent.choose_action(state, neighbors)

        edge_data = G_temp.get_edge_data(state, action)
        key = list(edge_data.keys())[0]

        cost = edge_data[key].get("dynamic_time", 1)
        risk = edge_data[key].get("risk", 0)

        # Reward (IMPORTANT)
        reward = -cost - 5 * risk

        next_state = action
        next_neighbors = list(G_temp.neighbors(next_state))

        agent.update(state, action, reward, next_state, next_neighbors)

        state = next_state

        if state == goal:
            break

    if ep % 20 == 0:
        print("Episode:", ep)

# Save Q-table
import pickle
pickle.dump(agent.q_table, open("q_table.pkl", "wb"))

print("✅ RL trained")