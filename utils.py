import osmnx as ox

def get_nearest_nodes(G, slat, slon, elat, elon):
    source = ox.distance.nearest_nodes(G, slon, slat)
    target = ox.distance.nearest_nodes(G, elon, elat)
    return source, target