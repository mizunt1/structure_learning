import pickle
import os

path = os.path.expanduser('~/projects/gflownet_sl-main/graph.pkl')
pickled = open(path, "rb")
graph = pickle.load(pickled)
a = graph.get_edge_data('B', 'D')
b = graph.get_edge_data('D', 'B')
c = graph.get_edge_data('E', 'D')
import pdb
pdb.set_trace()
