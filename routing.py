import networkx as nx

def get_path(G, source, target, weight):

    path = nx.shortest_path(G, source, target, weight=weight)
    cost = nx.shortest_path_length(G, source, target, weight=weight)

    return path, cost