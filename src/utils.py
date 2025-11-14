import numpy as np


def read_graph_from_file(filepath):
    # Đọc đồ thị từ file testcase theo định dạng:
    # Dòng 1: n_nodes n_edges
    # Các dòng sau: u v (1-based indices)
    # Trả về số đỉnh, số cạnh, danh sách cạnh (0-based), và ma trận kề adjacency
    with open(filepath, "r") as f:
        lines = f.readlines()
        n_nodes, n_edges = map(int, lines[0].split())
        edges = [(int(u) - 1, int(v) - 1) for u, v in (line.split() for line in lines[1:])]
    
    adjacency = np.zeros((n_nodes, n_nodes), dtype=int)
    for u, v in edges:
        adjacency[u, v] = 1
        adjacency[v, u] = 1
        
    return n_nodes, n_edges, edges, adjacency