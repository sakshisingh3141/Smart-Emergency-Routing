import pickle

q_table = pickle.load(open("q_table.pkl", "rb"))

def get_q(state, action):
    return q_table.get((state, action), 0.0)


def rl_route(G, source, target):

    path = [source]
    current = source
    visited = set()

    for _ in range(100):

        if current == target:
            break

        neighbors = list(G.neighbors(current))
        if not neighbors:
            break

        best = max(neighbors, key=lambda n: get_q(current, n))

        if best in visited:
            break

        visited.add(best)
        path.append(best)

        current = best

    return path