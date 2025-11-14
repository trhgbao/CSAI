import numpy as np

class Graph:
    """
    Đọc và lưu trữ đồ thị từ file
    Format file: dòng đầu là số đỉnh, các dòng sau là các cạnh (u v)
    """

    def __init__(self, file_path):
        self.file_path = file_path
        with open(file_path, "r") as f:
            lines = f.readlines()
            n_nodes, n_edges = map(int, lines[0].split())
            edges = [(int(u) - 1, int(v) - 1) for u, v in (line.split() for line in lines[1:])]
        
        adjacency = np.zeros((n_nodes, n_nodes), dtype=int)
        for u, v in edges:
            adjacency[u, v] = 1
            adjacency[v, u] = 1
        
        degrees = np.sum(adjacency, axis=1)
        self.max_degree = np.max(degrees)
            
        self.num_vertices = n_nodes
        self.adjacency = adjacency
        self.num_edges = n_edges
        self.max_colors = self.max_degree + 1

    def print_info(self):
        """In thông tin cơ bản của đồ thị"""
        print(f"--- Graph Info ---")
        print(f"Vertices: {self.num_vertices}")
        print(f"Edges: {self.num_edges}")
        print(f"Max Degree: {self.max_degree}")
        print(f"Color Upper Bound: {self.max_colors}")
        print(f"{'=' * 50}")