# problems/graph_definitions.py
import networkx as nx
import os

def load_graphs_from_files():
    """
    Quét thư mục 'test_cases', đọc các file .txt và tạo đồ thị.
    Các đồ thị input là 1-based.  Đồ thị trả về sẽ được đánh số từ 0.
    """
    graphs = {}
    current_dir = os.path.dirname(__file__)
    test_case_dir = os.path.join(current_dir, 'test_cases')

    if not os.path.isdir(test_case_dir):
        print(f"Cảnh báo: Không tìm thấy thư mục '{test_case_dir}'")
        return graphs

    for filename in os.listdir(test_case_dir):
        if filename.endswith(".txt"):
            graph = nx.Graph()
            file_path = os.path.join(test_case_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    # Đọc dòng đầu tiên để biết số đỉnh tổng thể
                    # Điều này hữu ích để thêm các đỉnh bị cô lập
                    num_vertices, _ = map(int, f.readline().split())
                    for i in range(num_vertices):
                        graph.add_node(i) # Thêm tất cả các đỉnh trước

                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            u, v = int(parts[0]) - 1, int(parts[1]) - 1
                            graph.add_edge(u, v)

                graph_normalized = nx.convert_node_labels_to_integers(
                    graph, first_label=0, ordering='default'
                )
                graphs[filename] = graph_normalized
                
            except Exception as e:
                print(f"Lỗi khi đọc file {filename}: {e}")

    return graphs

def get_graph_coloring_fitness(graph, colors):
    """
    Hàm đánh giá chất lượng của một lời giải tô màu bằng cách đếm số conflict.
    """
    conflicts = 0
    for u, v in graph.edges():
        if colors[u] == colors[v]:
            conflicts += 1
    return conflicts