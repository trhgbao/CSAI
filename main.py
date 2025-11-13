# main_app.py
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import networkx as nx

# --- Import các module bài toán và thuật toán ---
from problems.sphere_function import sphere
from problems.graph_definitions import load_graphs_from_files, get_graph_coloring_fitness
# from algorithms.pso_solver import PSO 
from algorithms.cs_mcoa_solver import StandardCS, ModifiedCS, StandardCS_DSATUR, ModifiedCS_DSATUR
from algorithms.traditional_solvers import (
    HillClimbingGraphColoring, SimulatedAnnealingGraphColoring, GeneticAlgorithmGraphColoring,
    HillClimbingSphere, SimulatedAnnealingSphere, GeneticAlgorithmSphere # Thêm các lớp mới
)
from algorithms.pso_solver import ParticleSwarmOptimizationSphere # Đã sửa tên lớp PSO
from algorithms.pso_gc_solver import PSOGraphColoring
from algorithms.aco_solver import ACO_GraphColoring
from algorithms.caco_solver import ContinuousACOSphere
from algorithms.abc_solver import ArtificialBeeColonyGraphColoring, ArtificialBeeColonySphere 
from algorithms.fa_solver import FireflyAlgorithmSphere, FireflyAlgorithmGraphColoring
from algorithms.fa_gc_solver import FireflyAlgorithmGraphColoring

class AlgorithmVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Trình Trực quan hóa Thuật toán Tối ưu")
        self.root.geometry("1400x1100")

        self.animation = None
        self.algorithm_runner = None

        self.comparison_runs = {
            "Sphere Function": [],
            "Graph Coloring": []
        }

        self.selected_comparison_run_data = None

        self.current_run_animation_data = [] 
        # --- MỚI: Các đối tượng FuncAnimation cho chế độ so sánh song song ---
        self.parallel_animation1 = None
        self.parallel_animation2 = None 

        # --- Cấu trúc Layout chính ---
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Bảng điều khiển (Trái)
        control_panel = tk.Frame(main_frame, width=300, relief=tk.RIDGE, borderwidth=2)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Khu vực trực quan hóa (Phải)
        self.vis_panel = tk.Frame(main_frame, relief=tk.RIDGE, borderwidth=2)
        self.vis_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._create_visualization_canvas()

        self._create_control_widgets(control_panel)

        # --- Xây dựng các thành phần ---
        
        self._on_problem_selected()
        

    def _create_control_widgets(self, parent):
        """Tạo các widget trong bảng điều khiển."""
        # 1. Lựa chọn bài toán
        problem_frame = ttk.LabelFrame(parent, text="1. Chọn Bài toán")
        problem_frame.pack(fill=tk.X, padx=5, pady=5)
        self.problem_var = tk.StringVar(value="Sphere Function")
        
        ttk.Radiobutton(problem_frame, text="Hàm Sphere (Liên tục)", variable=self.problem_var, 
                        value="Sphere Function", command=self._on_problem_selected).pack(anchor=tk.W)
        ttk.Radiobutton(problem_frame, text="Tô màu Đồ thị (Rời rạc)", variable=self.problem_var, 
                        value="Graph Coloring", command=self._on_problem_selected).pack(anchor=tk.W)

        # 2. Lựa chọn thuật toán
        algo_frame = ttk.LabelFrame(parent, text="2. Chọn Thuật toán")
        algo_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.algo_var = tk.StringVar()
        self.algo_dropdown = ttk.Combobox(algo_frame, textvariable=self.algo_var, state="readonly")
        self.algo_dropdown.pack(fill=tk.X)
        self.algo_dropdown.bind("<<ComboboxSelected>>", self._on_algorithm_selected)

        # 3. Khu vực tham số động
        self.param_frame = ttk.LabelFrame(parent, text="3. Tham số Thuật toán")
        self.param_frame.pack(fill=tk.X, padx=5, pady=5)
        self.param_widgets = {} # Lưu các widget tham số

        # 4. Khu vực tùy chọn bài toán
        self.problem_specific_frame = ttk.LabelFrame(parent, text="4. Tùy chọn Bài toán")
        self.problem_specific_frame.pack(fill=tk.X, padx=5, pady=5)
        self.problem_specific_widgets = {}

        # 5. Nút điều khiển
        control_frame = ttk.LabelFrame(parent, text="5. Điều khiển")
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        self.start_button = ttk.Button(control_frame, text="Bắt đầu", command=self.start_visualization)
        self.start_button.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.pause_button = ttk.Button(control_frame, text="Tạm dừng", command=self.pause_visualization, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.reset_button = ttk.Button(control_frame, text="Đặt lại", command=self.reset_visualization, state=tk.DISABLED)
        self.reset_button.pack(side=tk.LEFT, expand=True, fill=tk.X)

        comparison_control_frame = ttk.LabelFrame(parent, text="6. Quản lý & So sánh Runs")
        comparison_control_frame.pack(fill=tk.X, padx=5, pady=5)

        self.save_result_button = ttk.Button(comparison_control_frame, text="Lưu kết quả", command=self.save_current_run_for_comparison, state=tk.DISABLED)
        self.save_result_button.pack(fill=tk.X, pady=2)
        
        ttk.Label(comparison_control_frame, text="Danh sách các bản lưu:").pack(anchor=tk.W, pady=(5,0))
        self.saved_runs_listbox = tk.Listbox(comparison_control_frame, height=5, selectmode=tk.SINGLE)
        self.saved_runs_listbox.pack(fill=tk.X, expand=True)
        self.saved_runs_listbox.bind("<<ListboxSelect>>", self._on_saved_run_selected) # Gắn sự kiện chọn

        # Nút xóa bản lưu đã chọn
        self.delete_selected_button = ttk.Button(comparison_control_frame, text="Xóa bản lưu đã chọn", command=self._delete_selected_comparison_run, state=tk.DISABLED)
        self.delete_selected_button.pack(fill=tk.X, pady=2)

        self.compare_runs_button = ttk.Button(comparison_control_frame, text="Hiển thị so sánh hội tụ", command=self.compare_all_saved_runs, state=tk.DISABLED)
        self.compare_runs_button.pack(fill=tk.X, pady=2)

        self.clear_all_comparisons_button = ttk.Button(comparison_control_frame, text="Xóa TẤT CẢ bản lưu", command=self._clear_all_comparisons)
        self.clear_all_comparisons_button.pack(fill=tk.X, pady=2)
        
        # --- MỤC MỚI: Trực quan hóa song song ---
        side_by_side_frame = ttk.LabelFrame(parent, text="7. Trực quan hóa song song")
        side_by_side_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(side_by_side_frame, text="Chọn Run 1:").pack(anchor=tk.W)
        self.side_by_side_combo1 = ttk.Combobox(side_by_side_frame, state="readonly")
        self.side_by_side_combo1.pack(fill=tk.X, pady=2)
        self.side_by_side_combo1.bind("<<ComboboxSelected>>", lambda e: self._check_parallel_animation_buttons())

        ttk.Label(side_by_side_frame, text="Chọn Run 2:").pack(anchor=tk.W, pady=(5,0))
        self.side_by_side_combo2 = ttk.Combobox(side_by_side_frame, state="readonly")
        self.side_by_side_combo2.pack(fill=tk.X, pady=2)
        self.side_by_side_combo2.bind("<<ComboboxSelected>>", lambda e: self._check_parallel_animation_buttons())

        self.run_parallel_animation_button = ttk.Button(side_by_side_frame, text="Chạy minh họa song song", command=self._start_parallel_visualization, state=tk.DISABLED)
        self.run_parallel_animation_button.pack(fill=tk.X, pady=2)

        self.stop_parallel_animation_button = ttk.Button(side_by_side_frame, text="Dừng minh họa song song", command=self._stop_parallel_visualization, state=tk.DISABLED)
        self.stop_parallel_animation_button.pack(fill=tk.X, pady=2)

        # --- MỚI: Nút xóa hình minh họa song song ---
        self.clear_parallel_vis_button = ttk.Button(side_by_side_frame, text="Xóa hình minh họa song song", command=self._clear_side_by_side_visualizations)
        self.clear_parallel_vis_button.pack(fill=tk.X, pady=2)
        # Tải cấu hình ban đầu
        # self._on_problem_selected()

    def _create_dynamic_parameters(self, algorithm_name):
        """Tạo các ô nhập liệu tham số dựa trên thuật toán được chọn."""
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        self.param_widgets.clear()

        problem_name = self.problem_var.get().strip()
        
        # Khởi tạo các dictionary
        common_params = {}
        specific_params = {}

        if problem_name == "Graph Coloring":
            common_params = {
                "N Iter": "200",
                "N Pop": "50",
                "Penalty Weight": "1000"
            }
            
            if algorithm_name == "Cuckoo Search":
                specific_params = {"P Abandon": "0.25"}
            elif algorithm_name == "Modified Cuckoo Search (MCOA)":
                specific_params = {"P Abandon": "0.25", "Local Search Steps": "50"}
            elif algorithm_name == "CS + DSATUR":
                specific_params = {"P Abandon": "0.25", "DSATUR Ratio": "0.8"}
            elif algorithm_name == "MCOA + DSATUR":
                specific_params = {"P Abandon": "0.25", "Local Search Steps": "50", "DSATUR Ratio": "0.8"}
            elif algorithm_name == "Hill Climbing":
                del common_params["N Pop"] 
                common_params["N Iter"] = "2000"
            elif algorithm_name == "Simulated Annealing":
                del common_params["N Pop"]
                common_params["N Iter"] = "5000"
                specific_params = {"Initial Temp": "1000", "Cooling Rate": "0.99"}
            elif algorithm_name == "Genetic Algorithm":
                specific_params = {"Mutation Rate": "0.05", "Crossover Rate": "0.8"}
            elif algorithm_name == "ACO (DSATUR)" or algorithm_name == "ACO":
                common_params["N Pop"] = "40"
                specific_params = { "Alpha": "1.66", "Beta": "0.8", "Rho (Evaporation)": "0.35",
                                    "Q (Pheromone)": "100.0", "Gamma": "1e6"}
            elif algorithm_name == "Artificial Bee Colony":
                specific_params = {"Limit": "100"}
            elif algorithm_name == "Firefly Algorithm":
                common_params["N Pop"] = "40"
                specific_params = { "Alpha0 (Initial)": "0.5", "Alpha Min": "0.01", "Alpha Decay": "0.97",
                                    "Beta Max": "1.0", "Beta Min": "0.2", "Gamma": "0.1", 
                                    "P_local (Prob)": "0.3", "I_local (Iter)": "10" }

        elif problem_name == "Sphere Function":
            common_params = {
                "N Iter": "1000",
                "N Pop": "50"
            }

            if algorithm_name == "PSO":
                specific_params = {"C1": "1.5", "C2": "1.5", "W": "0.7"}
            elif algorithm_name == "Firefly Algorithm":
                common_params["N Pop"] = "40"
                specific_params = { "Alpha0 (Initial)": "0.5", "Alpha Min": "1e-9", "Alpha Decay": "0.99",
                                    "Beta0": "1.0", "Gamma": "0.001", "P_local (Prob)": "0.3", 
                                    "I_local (Iter)": "10" }
            elif algorithm_name == "Artificial Bee Colony":
                specific_params = {"Limit": "100"}
            elif algorithm_name == "Genetic Algorithm":
                common_params["N Iter"] = "200"
                specific_params = {"Mutation Rate": "0.05", "Crossover Rate": "0.8"}
            elif algorithm_name == "Simulated Annealing":
                del common_params["N Pop"]
                common_params["N Iter"] = "5000"
                specific_params = {"Initial Temp": "1000", "Cooling Rate": "0.99"}
            elif algorithm_name == "Hill Climbing":
                del common_params["N Pop"]
                common_params["N Iter"] = "2000"
                specific_params = {"Step Size": "0.05"}
            elif algorithm_name == "Continuous ACO":
                common_params["N Pop"] = "30" # N Ants
                specific_params = { "N Archive": "30", "Q (Std Dev)": "0.05", "Xi (Evaporation)": "0.85" }

        # --- TẠO WIDGET TỪ CÁC DICTIONARY ĐÃ TỔNG HỢP ---
        all_params = {**common_params, **specific_params}
        
        for name, default_value in all_params.items():
            row = tk.Frame(self.param_frame)
            row.pack(fill=tk.X, pady=2)
            label = ttk.Label(row, text=name, width=25)
            label.pack(side=tk.LEFT)
            entry = ttk.Entry(row)
            entry.pack(side=tk.RIGHT, expand=True, fill=tk.X)
            entry.insert(0, default_value)
            self.param_widgets[name] = entry

    def _create_problem_specific_options(self, problem_name):
        """Tạo các tùy chọn cho từng bài toán cụ thể."""
        for widget in self.problem_specific_frame.winfo_children():
            widget.destroy()
        self.problem_specific_widgets.clear()

        if problem_name == "Sphere Function":
            # Dimension
            row_d = tk.Frame(self.problem_specific_frame)
            row_d.pack(fill=tk.X, pady=2)
            ttk.Label(row_d, text="Số chiều (D)", width=15).pack(side=tk.LEFT)
            entry_d = ttk.Entry(row_d)
            entry_d.pack(side=tk.RIGHT, expand=True, fill=tk.X)
            entry_d.insert(0, "2")
            self.problem_specific_widgets["Dimension"] = entry_d
            
            # Range
            row_r = tk.Frame(self.problem_specific_frame)
            row_r.pack(fill=tk.X, pady=2)
            ttk.Label(row_r, text="Phạm vi (min,max)", width=15).pack(side=tk.LEFT)
            entry_r = ttk.Entry(row_r)
            entry_r.pack(side=tk.RIGHT, expand=True, fill=tk.X)
            entry_r.insert(0, "-5.12, 5.12")
            self.problem_specific_widgets["Range"] = entry_r

        elif problem_name == "Graph Coloring":
            self.graphs = load_graphs_from_files()
            row = tk.Frame(self.problem_specific_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text="Test Case", width=15).pack(side=tk.LEFT)
            combo = ttk.Combobox(row, values=list(self.graphs.keys()), state="readonly")
            combo.pack(side=tk.RIGHT, expand=True, fill=tk.X)
            combo.current(0)
            self.problem_specific_widgets["Test Case"] = combo

    def _on_problem_selected(self):
        """Xử lý khi người dùng chọn một bài toán mới."""
        problem = self.problem_var.get()
        if problem == "Sphere Function":
            self.algo_dropdown['values'] = [
                "PSO", 
                "Firefly Algorithm",
                "Artificial Bee Colony",
                "Genetic Algorithm",
                "Simulated Annealing",
                "Hill Climbing",
                "Continuous ACO"
            ]
        elif problem == "Graph Coloring":
            self.algo_dropdown['values'] = [
                "Cuckoo Search", 
                "Modified Cuckoo Search (MCOA)", 
                "CS + DSATUR",
                "PSO (Real)", 
                "MCOA + DSATUR",
                "Hill Climbing",
                "Simulated Annealing",
                "Genetic Algorithm",
                "ACO (DSATUR)",
                "Artificial Bee Colony",
                "Firefly Algorithm"
            ]
        
        # Đảm bảo có ít nhất 1 thuật toán được chọn
        if self.algo_dropdown['values']:
            self.algo_dropdown.current(0)
        else:
            self.algo_dropdown.set('') # Xóa trống nếu không có thuật toán nào

        self._create_problem_specific_options(problem)
        self._on_algorithm_selected(event=None) # Cập nhật tham số theo thuật toán mặc định

        self._update_saved_runs_listbox() # Cập nhật listbox và nút khi đổi bài toán
        self.compare_all_saved_runs(clear_only=True) # Xóa đồ thị hội tụ khi đổi bài toán
        self._clear_side_by_side_visualizations() # Xóa hiển thị side-by-side
        self._stop_parallel_visualization() # Đảm bảo mọi animation song song bị dừng

        self.start_button.config(state=tk.NORMAL) 
        self.pause_button.config(state=tk.DISABLED)
        self.reset_button.config(state=tk.DISABLED)

    def _on_algorithm_selected(self, event):
        """Xử lý khi người dùng chọn một thuật toán mới."""
        self._create_dynamic_parameters(self.algo_var.get())

        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.reset_button.config(state=tk.DISABLED)

    def _create_visualization_canvas(self):
        """Tạo canvas Matplotlib để vẽ."""
        self.fig = Figure(figsize=(10, 10), dpi=100) # Tăng kích thước fig tổng thể

        # GridSpec để quản lý layout phức tạp hơn: 3 hàng, 2 cột
        # height_ratios: Live Vis (1), Saved Vis (1), Convergence (0.8)
        gs = self.fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8])  

        self.ax_main = self.fig.add_subplot(gs[0, :]) # Hàng 0, chiếm cả 2 cột (Live Visualization)
        self.ax_comp1 = self.fig.add_subplot(gs[1, 0]) # Hàng 1, cột 0 (Saved Run 1)
        self.ax_comp2 = self.fig.add_subplot(gs[1, 1]) # Hàng 1, cột 1 (Saved Run 2)
        self.ax_conv = self.fig.add_subplot(gs[2, :]) # Hàng 2, chiếm cả 2 cột (Convergence Plot)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.vis_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


    def start_visualization(self):
        """Bắt đầu quá trình trực quan hóa."""
        is_ready = self.setup_visualization()
        if not is_ready:
            messagebox.showwarning("Chưa Cài đặt", f"Thuật toán '{self.algo_var.get()}' chưa được tích hợp hoặc có lỗi.")
            self.reset_visualization()
            return

        # --- Reset các biến trạng thái quan trọng ---
        self.animation_finished = False # Cờ báo hiệu kết thúc
        self.start_time = time.time() # Bắt đầu đo thời gian
        
        self.start_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.reset_button.config(state=tk.NORMAL)

        if self.animation is None:
            num_frames = int(self.param_widgets["N Iter"].get())
            
            self.animation = FuncAnimation(
                self.fig,
                self.update_frame,
                frames=num_frames,
                interval=1,
                repeat=False,
                blit=False
            )
            self.canvas.draw()
        else:
            # Nếu đang resume, không cần làm gì thêm ở đây
            self.animation.resume()

    def setup_visualization(self):
        """Khởi tạo môi trường trực quan hóa dựa trên lựa chọn."""
        self.ax_main.clear()
        self.ax_conv.clear()

        problem = self.problem_var.get().strip()
        algorithm = self.algo_var.get().strip()

        is_implemented = False
        
        if problem == "Sphere Function":
            params = self._get_params()
            try:
                bounds_str = self.problem_specific_widgets["Range"].get().split(',')
                # Đảm bảo bounds là một tuple chứa min và max
                # Hoặc một mảng 2D cho mỗi chiều nếu dims > 1 (tùy thuộc vào cách solver mong đợi)
                self.bounds_min = float(bounds_str[0])
                self.bounds_max = float(bounds_str[1])
                self.dims = int(self.problem_specific_widgets["Dimension"].get())

                # Đối với Sphere function và các thuật toán liên tục, bounds thường là (min, max) cho mỗi chiều
                # Nếu dims = 2, bounds sẽ là ([min, min], [max, max]) hoặc đơn giản là (min, max) và hàm xử lý
                # Hiện tại, pso_solver mong đợi `bounds` là một tuple (min_val, max_val) và tính n_dims
                # Nhưng các solver khác như GeneticAlgorithmSphere lại mong đợi `bounds=(min_bound, max_bound)`
                # Dựa vào pso_solver.py: `self.n_dims = len(bounds[0]) if hasattr(bounds[0], '__len__') else 2`
                # Nếu bounds là (min_val, max_val) thì bounds[0] là min_val (float), không có len().
                # Nó sẽ mặc định dims=2.
                # Nếu bạn muốn truyền dims khác 2, bạn cần làm bounds trở thành một mảng của các cặp.
                # Ví dụ: bounds=np.array([[min,max], [min,max], ...])
                # Để giữ cho nó nhất quán với các solver khác, hãy truyền nó dưới dạng một tuple (min, max)
                # và để solver tự xác định số chiều.

                # Cập nhật để tạo bounds theo định dạng phù hợp với các solver của bạn.
                # Các solver như ParticleSwarmOptimizationSphere, FireflyAlgorithmSphere,
                # ArtificialBeeColonySphere, SimulatedAnnealingSphere, HillClimbingSphere
                # đều nhận `bounds=(self.bounds[0], self.bounds[1])` và tính `n_dims` hoặc nhận `dims` trực tiếp.

                # Đảm bảo self.bounds được định nghĩa một cách nhất quán
                self.bounds = (self.bounds_min, self.bounds_max) # Tuple (min, max)

            except (ValueError, IndexError):
                messagebox.showerror("Lỗi Tham số", "Phạm vi hoặc Số chiều không hợp lệ.")
                return False
            
            # Khởi tạo thuật toán
            if algorithm == "PSO":
                self.algorithm_runner = ParticleSwarmOptimizationSphere(
                    objective_func=sphere,
                    bounds=self.bounds, # Truyền tuple (min, max)
                    n_particles=int(params["N Pop"]),
                    n_iter=int(params["N Iter"]),
                    # n_dims=self.dims, # <--- Xóa dòng này
                    c1=float(params["C1"]),
                    c2=float(params["C2"]),
                    w=float(params["W"])
                )
                is_implemented = True
            elif algorithm == "Firefly Algorithm":
                self.algorithm_runner = FireflyAlgorithmSphere(
                    objective_func=sphere,
                    bounds=(self.bounds[0], self.bounds[1]),
                    n_pop=int(params["N Pop"]),
                    n_iter=int(params["N Iter"]),
                    alpha0=float(params["Alpha0 (Initial)"]),
                    alpha_min=float(params["Alpha Min"]),
                    alpha_decay=float(params["Alpha Decay"]),
                    beta0=float(params["Beta0"]),
                    gamma=float(params["Gamma"]),
                    p_local=float(params["P_local (Prob)"]), # Tham số mới
                    i_local=int(params["I_local (Iter)"])   # Tham số mới
                )
                is_implemented = True
            elif algorithm == "Continuous ACO":
                self.algorithm_runner = ContinuousACOSphere(
                    objective_func=sphere,
                    bounds=self.bounds,
                    dims=self.dims, # Caco_solver nhận dims trực tiếp
                    n_ants=int(params["N Pop"]),
                    n_archive=int(params["N Archive"]),
                    q=float(params["Q (Std Dev)"]),
                    xi=float(params["Xi (Evaporation)"]),
                    n_iter=int(params["N Iter"])
                )
                is_implemented = True
            elif algorithm == "Artificial Bee Colony":
                self.algorithm_runner = ArtificialBeeColonySphere(
                    objective_func=sphere,
                    bounds=self.bounds,
                    n_pop=int(params["N Pop"]),
                    n_iter=int(params["N Iter"]),
                    limit=int(params["Limit"])
                )
                is_implemented = True
            elif algorithm == "Genetic Algorithm":
                self.algorithm_runner = GeneticAlgorithmSphere(
                    objective_func=sphere,
                    bounds=self.bounds,
                    dims=self.dims, # GeneticAlgorithmSphere nhận dims trực tiếp
                    n_pop=int(params["N Pop"]),
                    n_iter=int(params["N Iter"]),
                    mutation_rate=float(params["Mutation Rate"]),
                    crossover_rate=float(params["Crossover Rate"])
                )
                is_implemented = True
            elif algorithm == "Simulated Annealing":
                self.algorithm_runner = SimulatedAnnealingSphere(
                    objective_func=sphere,
                    bounds=self.bounds,
                    dims=self.dims, # SimulatedAnnealingSphere nhận dims trực tiếp
                    n_iter=int(params["N Iter"]),
                    initial_temp=float(params["Initial Temp"]),
                    cooling_rate=float(params["Cooling Rate"]),
                )
                is_implemented = True
            elif algorithm == "Hill Climbing":
                self.algorithm_runner = HillClimbingSphere(
                    objective_func=sphere,
                    bounds=self.bounds,
                    dims=self.dims, # HillClimbingSphere nhận dims trực tiếp
                    n_iter=int(params["N Iter"]),
                    step_size=float(params["Step Size"])
                    
                )
                is_implemented = True
            
            if is_implemented:
                self._setup_sphere_vis()

        elif problem == "Graph Coloring":
            self._setup_graph_vis() # Phải gọi trước để current_graph có giá trị
            params = self._get_params()

            if algorithm == "Cuckoo Search":
                self.algorithm_runner = StandardCS(
                    graph=self.current_graph,
                    n_pop=int(params["N Pop"]),
                    p_abandon=float(params["P Abandon"]),
                    penalty_weight=int(params["Penalty Weight"])
                )
                is_implemented = True
                
            elif algorithm == "Modified Cuckoo Search (MCOA)":
                self.algorithm_runner = ModifiedCS(
                    graph=self.current_graph,
                    n_pop=int(params["N Pop"]),
                    p_abandon=float(params["P Abandon"]),
                    penalty_weight=int(params["Penalty Weight"]),
                    local_search_steps=int(params["Local Search Steps"])
                )
                is_implemented = True

            elif algorithm == "CS + DSATUR":
                self.algorithm_runner = StandardCS_DSATUR(
                    graph=self.current_graph,
                    n_pop=int(params["N Pop"]),
                    p_abandon=float(params["P Abandon"]),
                    penalty_weight=int(params["Penalty Weight"]),
                    dsatur_ratio=float(params["DSATUR Ratio"])
                )
                is_implemented = True

            elif algorithm == "MCOA + DSATUR":
                self.algorithm_runner = ModifiedCS_DSATUR(
                    graph=self.current_graph,
                    n_pop=int(params["N Pop"]),
                    p_abandon=float(params["P Abandon"]),
                    penalty_weight=int(params["Penalty Weight"]),
                    local_search_steps=int(params["Local Search Steps"]),
                    dsatur_ratio=float(params["DSATUR Ratio"])
                )
                is_implemented = True

            elif algorithm == "Hill Climbing":
                self.algorithm_runner = HillClimbingGraphColoring(
                    graph=self.current_graph,
                    penalty_weight=int(params["Penalty Weight"])
                )
                is_implemented = True

            elif algorithm == "Simulated Annealing":
                self.algorithm_runner = SimulatedAnnealingGraphColoring(
                    graph=self.current_graph,
                    penalty_weight=int(params["Penalty Weight"]),
                    initial_temp=float(params["Initial Temp"]),
                    cooling_rate=float(params["Cooling Rate"])
                )
                is_implemented = True

            elif algorithm == "Genetic Algorithm":
                self.algorithm_runner = GeneticAlgorithmGraphColoring(
                    graph=self.current_graph,
                    penalty_weight=int(params["Penalty Weight"]),
                    n_pop=int(params["N Pop"]),
                    mutation_rate=float(params["Mutation Rate"]),
                    crossover_rate=float(params["Crossover Rate"])
                )
                is_implemented = True
            elif algorithm == "Artificial Bee Colony":
                self.algorithm_runner = ArtificialBeeColonyGraphColoring(
                    graph=self.current_graph,
                    penalty_weight=int(params["Penalty Weight"]),
                    n_pop=int(params["N Pop"]),
                    n_iter=int(params["N Iter"]), # n_iter không dùng trong init của ABC solver, nhưng giữ để đồng bộ
                    limit=int(params["Limit"])
                )
                is_implemented = True
            elif algorithm == "ACO (DSATUR)":
                # ACO solver cần n_colors ban đầu, có thể ước tính từ bậc của đồ thị
                degrees = [d for n, d in self.current_graph.degree()]
                initial_colors_estimate = max(degrees) + 1 if degrees else 1

                self.algorithm_runner = ACO_GraphColoring(
                    graph=self.current_graph,
                    n_colors=initial_colors_estimate, # Số màu tối đa cho kiến chọn
                    n_ants=int(params["N Pop"]),
                    n_iter=int(params["N Iter"]), # n_iter không dùng trong init của ACO solver, nhưng giữ để đồng bộ
                    alpha=float(params["Alpha"]),
                    beta=float(params["Beta"]),
                    rho=float(params["Rho (Evaporation)"]),
                    q=float(params["Q (Pheromone)"]),
                    use_dsatur=True,
                    gamma=float(params["Gamma"]),
                    # penalty_weight=int(params["Penalty Weight"])
                )
                is_implemented = True
            elif algorithm == "ACO":
                # ACO solver cần n_colors ban đầu, có thể ước tính từ bậc của đồ thị
                degrees = [d for n, d in self.current_graph.degree()]
                initial_colors_estimate = max(degrees) + 1 if degrees else 1

                self.algorithm_runner = ACO_GraphColoring(
                    graph=self.current_graph,
                    n_colors=initial_colors_estimate, # Số màu tối đa cho kiến chọn
                    n_ants=int(params["N Pop"]),
                    n_iter=int(params["N Iter"]), # n_iter không dùng trong init của ACO solver, nhưng giữ để đồng bộ
                    alpha=float(params["Alpha"]),
                    beta=float(params["Beta"]),
                    rho=float(params["Rho (Evaporation)"]),
                    q=float(params["Q (Pheromone)"]),
                    use_dsatur=False,
                    gamma=float(params["Gamma"]),
                    # penalty_weight=int(params["Penalty Weight"])
                )
                is_implemented = True
            elif algorithm == "Firefly Algorithm":
                self.algorithm_runner = FireflyAlgorithmGraphColoring(
                    graph=self.current_graph,
                    n_pop=int(params["N Pop"]),
                    n_iter=int(params["N Iter"]),
                    penalty_weight=int(params["Penalty Weight"]),
                    alpha0=float(params["Alpha0 (Initial)"]),
                    alpha_min=float(params["Alpha Min"]),
                    alpha_decay=float(params["Alpha Decay"]),
                    beta_max=float(params["Beta Max"]), # Tham số mới
                    beta_min=float(params["Beta Min"]), # Tham số mới
                    gamma=float(params["Gamma"]),
                    p_local=float(params["P_local (Prob)"]),
                    i_local=int(params["I_local (Iter)"])
                )
                is_implemented = True

            elif algorithm == "PSO (Real)":
                degrees = [d for n, d in self.current_graph.degree()]
                max_color = max(degrees) + 1 if degrees else 1
                self.algorithm_runner = PSOGraphColoring(
                    graph=self.current_graph,
                    max_color=max_color,
                    n_pop=int(params["N Pop"]),
                    n_iter=int(params["N Iter"]),
                    w=float(params["W"]),
                    c1=float(params["C1"]),
                    c2=float(params["C2"]),
                    penalty_weight=int(params["Penalty Weight"])
                )
                is_implemented = True

        return is_implemented

    def _setup_sphere_vis(self):
        """Cài đặt cho trực quan hóa hàm Sphere."""
        self.ax_main.clear()
        self.ax_conv.clear()

        # --- Cài đặt đồ thị hội tụ (luôn luôn) ---
        self.ax_conv.set_title("Lịch sử hội tụ (Best Fitness)")
        self.ax_conv.set_xlabel("Thế hệ (Iteration)")
        self.ax_conv.set_ylabel("Giá trị Fitness tốt nhất (Log Scale)")
        # DÒNG QUAN TRỌNG: KHỞI TẠO LẠI HOÀN TOÀN self.conv_line
        self.conv_line, = self.ax_conv.plot([], []) 
        self.ax_conv.grid(True)

        # --- Cài đặt đồ thị chính (chỉ khi D=2) ---
        if self.dims == 2:
            # ... (code cũ để vẽ contour và scatter)
            min_b, max_b = self.bounds[0], self.bounds[1]
            x_coords, y_coords = np.linspace(min_b, max_b, 100), np.linspace(min_b, max_b, 100)
            X, Y = np.meshgrid(x_coords, y_coords)
            Z = X**2 + Y**2
            self.ax_main.contourf(X, Y, Z, levels=50, cmap='viridis')
            self.ax_main.set_title("Hàm Sphere (2D) và Quá trình Hội tụ", fontsize=10)
            self.ax_main.set_xlabel("x1")
            self.ax_main.set_ylabel("x2")

            initial_pos = np.zeros((1, 2))
            if self.algorithm_runner and hasattr(self.algorithm_runner, 'positions'):
                initial_pos = self.algorithm_runner.positions
            self.scatter = self.ax_main.scatter(initial_pos[:, 0], initial_pos[:, 1], c='red', s=10, zorder=10)
        else:
            self.ax_main.set_title(f'Trực quan hóa cho D={self.dims}', fontsize=10)
            self.ax_main.text(0.5, 0.5, f'Không thể vẽ không gian {self.dims} chiều.\nTheo dõi sự hội tụ ở biểu đồ bên dưới.',
                            ha='center', va='center', fontsize=12)

    # --- HÀM CHO GRAPH COLORING ---
    def _setup_graph_vis(self):
        """Cài đặt cho trực quan hóa tô màu đồ thị."""
        test_case_name = self.problem_specific_widgets["Test Case"].get()
        # ... (code cũ để tải đồ thị)

        self.current_graph = self.graphs[test_case_name]
        self.graph_pos = nx.spring_layout(self.current_graph, seed=42)

        # --- Phần thiết lập đồ thị chính ---
        self.ax_main.clear()
        nx.draw_networkx_edges(self.current_graph, self.graph_pos, ax=self.ax_main)
        self.node_collection = nx.draw_networkx_nodes(
            self.current_graph, self.graph_pos, node_color='grey', node_size=500, ax=self.ax_main
        )
        # nx.draw_networkx_labels(...) # Đã comment

        # --- Phần thiết lập đồ thị hội tụ ---
        self.ax_conv.clear()
        self.ax_conv.set_title("Lịch sử hội tụ (Best Fitness)")
        self.ax_conv.set_xlabel("Thế hệ (Iteration)")
        self.ax_conv.set_ylabel("Giá trị Fitness (Số màu + Phạt)")
        # DÒNG QUAN TRỌNG: KHỞI TẠO LẠI HOÀN TOÀN self.conv_line
        self.conv_line, = self.ax_conv.plot([], [])
        self.ax_conv.grid(True)

    def update_frame(self, frame_num):
        """Hàm cập nhật cho mỗi frame của animation."""
        if not self.algorithm_runner:
            return
        
        # --- LOGIC MỚI: TÍNH TOÁN VÀ HIỂN THỊ "LIVE" ---
        
        # Lấy trạng thái hiện tại từ thuật toán
        state = self.algorithm_runner.step()
        
        # Tính thời gian đã trôi qua
        elapsed_time = time.time() - self.start_time
        
        problem = self.problem_var.get().strip()

        # --- Cập nhật cho BÀI TOÁN SPHERE ---
        if problem == "Sphere Function":
            gbest_val = state.get("gbest_val", float('inf'))
            history = state.get("history", [])

            # Cập nhật tiêu đề đồ thị chính (ax_main)
            main_title = (f"Hàm Sphere (D={self.dims}) | "
                        f"Thời gian: {elapsed_time:.2f}s | "
                        f"Best: {gbest_val:.4e}")
            self.ax_main.set_title(main_title, fontsize=10)

            # Cập nhật đồ thị chính (chỉ khi D=2)
            if self.dims == 2:
                positions = state.get("positions")
                if positions is not None:
                    self.scatter.set_offsets(positions)

            # Cập nhật đồ thị hội tụ
            if history:
                self.conv_line.set_data(range(len(history)), history)
                self.ax_conv.relim()
                self.ax_conv.autoscale_view()
                self.ax_conv.set_yscale('log')

        # --- Cập nhật cho BÀI TOÁN TÔ MÀU ĐỒ THỊ ---
        elif problem == "Graph Coloring":
            colors = state.get("colors")
            fitness, num_colors, num_conflicts = state.get("fitness_tuple", (0, 0, 0))
            history = state.get("history", [])

            # Cập nhật tiêu đề đồ thị chính (ax_main)
            main_title = (f"Tô màu: {self.problem_specific_widgets['Test Case'].get()} | "
                        f"Thời gian: {elapsed_time:.2f}s\n"
                        f"Số màu: {num_colors} | Xung đột: {num_conflicts} | "
                        f"Fitness: {fitness:.2f}")
            self.ax_main.set_title(main_title, fontsize=10)

            # Cập nhật màu sắc các đỉnh
            if colors is not None and colors.size > 0:
                num_distinct_colors = np.max(colors) + 1
                cmap = plt.colormaps.get_cmap('viridis')
                palette = cmap(np.linspace(0, 1, num_distinct_colors))
                node_colors = [palette[c] for c in colors]
                self.node_collection.set_color(node_colors)
            
            # Cập nhật đồ thị hội tụ
            if history:
                self.conv_line.set_data(range(len(history)), history)
                self.ax_conv.relim()
                self.ax_conv.autoscale_view()
                self.ax_conv.set_ylim(bottom=0)
        
        # Vẽ lại canvas để hiển thị các thay đổi
        self.fig.canvas.draw_idle()

        # --- Logic kết thúc animation (vẫn giữ để cho phép chạy lại) ---
        num_frames = int(self.param_widgets["N Iter"].get())
        if frame_num >= num_frames - 1:
            if self.animation and self.animation.event_source:
                self.animation.event_source.stop()
            self.animation = None
            self.start_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.DISABLED)

    def pause_visualization(self):
        """Tạm dừng hoặc tiếp tục animation."""
        # Kiểm tra xem animation có đang chạy hay không trước khi tương tác
        if self.animation and self.animation.event_source is not None:
            if self.pause_button['text'] == "Tạm dừng":
                self.animation.pause()
                self.pause_button.config(text="Tiếp tục")
                self.start_button.config(state=tk.NORMAL, text="Tiếp tục") 
            else:
                self.animation.resume()
                self.pause_button.config(text="Tạm dừng")
                self.start_button.config(state=tk.DISABLED, text="Đang chạy...") 

    def reset_visualization(self):
        """Dừng và đặt lại toàn bộ quá trình."""
        self.reset_current_visualization_state() # Gọi hàm mới để dọn dẹp canvas

        # Các nút điều khiển
        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED, text="Tạm dừng")
        self.reset_button.config(state=tk.DISABLED) # Vẫn vô hiệu hóa reset nếu không cần thiết
        self.save_result_button.config(state=tk.DISABLED) 
        
        # Đặt lại các tùy chọn bài toán và thuật toán
        self._on_problem_selected() 
        self._stop_parallel_visualization() # Đảm bảo mọi animation song song bị dừng
        self._clear_side_by_side_visualizations() # Và xóa hiển thị

        self._on_problem_selected()

    def _get_params(self):
        """Lấy giá trị từ các ô nhập liệu tham số."""
        return {name: entry.get() for name, entry in self.param_widgets.items()}
    
    def save_current_run_for_comparison(self):
        """Lưu lại lịch sử hội tụ và tham số của lần chạy hiện tại để so sánh."""
        if not self.algorithm_runner:
            messagebox.showwarning("Lỗi", "Chưa có thuật toán nào được chạy.")
            return

        current_problem_name = self.problem_var.get()
        algo_name = self.algo_var.get()
        params = self._get_params()

        final_state = self.algorithm_runner._get_state()
        history = final_state["history"]

        if not history or (len(history) <= 1 and history[0] == float('inf')):
             messagebox.showwarning("Lỗi", "Lịch sử hội tụ trống hoặc chỉ có giá trị ban đầu. Thuật toán có thể chưa chạy hết hoặc không tìm thấy lời giải.")
             return

        label_parts = [algo_name]
        if current_problem_name == "Sphere Function":
            label_parts.append(f"D={self.dims}")
            if "N Pop" in params: label_parts.append(f"Pop={params['N Pop']}")
            if "C1" in params: label_parts.append(f"C1={params['C1']}")
            if "C2" in params: label_parts.append(f"C2={params['C2']}")
            if "Alpha" in params: label_parts.append(f"A={params['Alpha']}")
            if "Beta0" in params: label_parts.append(f"B0={params['Beta0']}")
            if "Gamma" in params: label_parts.append(f"G={params['Gamma']}")
            if "Limit" in params: label_parts.append(f"L={params['Limit']}")
        elif current_problem_name == "Graph Coloring":
            test_case_name = self.problem_specific_widgets["Test Case"].get()
            label_parts.append(f"Test={test_case_name}")
            if "N Pop" in params: label_parts.append(f"Pop={params['N Pop']}")
            if "P Abandon" in params: label_parts.append(f"P_ab={params['P Abandon']}")
            if "DSATUR Ratio" in params: label_parts.append(f"DSATUR={params['DSATUR Ratio']}")
            if "Local Search Steps" in params: label_parts.append(f"LS={params['Local Search Steps']}")
            if "Initial Temp" in params: label_parts.append(f"T0={params['Initial Temp']}")
            if "Cooling Rate" in params: label_parts.append(f"CR={params['Cooling Rate']}")
            if "Mutation Rate" in params: label_parts.append(f"MR={params['Mutation Rate']}")

        run_label = f"{current_problem_name} | {' '.join(label_parts)}"
        
        # Lấy trạng thái cuối cùng cho visualization chính
        final_main_state = {}
        if current_problem_name == "Sphere Function":
            final_main_state = {'positions': final_state['positions'].copy(), 'gbest_val': final_state['gbest_val'], 'bounds': self.bounds}
        elif current_problem_name == "Graph Coloring":
            # Cần lưu cả đồ thị và vị trí các đỉnh để vẽ lại
            final_main_state = {'colors': final_state['colors'].copy(), 'fitness_tuple': final_state['fitness_tuple'], 
                                'graph': self.current_graph, 'graph_pos': self.graph_pos} 

        if current_problem_name not in self.comparison_runs:
            self.comparison_runs[current_problem_name] = []

        self.comparison_runs[current_problem_name].append({
            'label': run_label,
            'history': list(history),
            'params': params, 
            'animation_data': list(self.current_run_animation_data) # LƯU DỮ LIỆU ANIMATION FRAME
        })
        messagebox.showinfo("Thông báo", f"Đã lưu kết quả của '{run_label}' để so sánh.")
        self.save_result_button.config(state=tk.DISABLED)
        self._update_saved_runs_listbox() 

    def compare_all_saved_runs(self, clear_only=False):
        """Vẽ tất cả các lịch sử hội tụ đã lưu trên cùng một đồ thị."""
        self.ax_conv.clear()
        if clear_only:
            self.canvas.draw()
            return # Chỉ xóa, không vẽ gì nếu chỉ muốn clear
        
        current_problem = self.problem_var.get() # Lấy bài toán hiện tại
        runs_to_compare = self.comparison_runs.get(current_problem, []) 

        if not runs_to_compare:
            messagebox.showinfo("Thông báo", f"Chưa có lần chạy nào được lưu cho bài toán '{current_problem}' để so sánh.")
            # Thiết lập lại các nhãn cho đồ thị trống
            self.ax_conv.set_title("Lịch sử hội tụ (Best Fitness)")
            self.ax_conv.set_xlabel("Lần lặp")
            self.ax_conv.set_ylabel("Giá trị Fitness (log)")
            self.ax_conv.grid(True)
            self.ax_conv.set_yscale('log')
            self.canvas.draw()
            return

        self.ax_conv.set_title(f"So sánh các lần chạy cho Bài toán: {current_problem}") # THAY ĐỔI TIÊU ĐỀ
        self.ax_conv.set_xlabel("Lần lặp")
        self.ax_conv.set_ylabel("Giá trị Fitness (log)")
        self.ax_conv.set_yscale('log')
        self.ax_conv.grid(True, which="both", ls="--")

        ccolors = plt.colormaps.get_cmap('tab10').resampled(max(1, len(runs_to_compare))) # Đảm bảo có ít nhất 1 màu

        max_len = 0
        for i, run_data in enumerate(runs_to_compare):
            # --- MỚI: Kiểm tra kiểu dữ liệu để tránh TypeError ---
            if not isinstance(run_data, dict):
                print(f"Cảnh báo: Bỏ qua kiểu dữ liệu không mong muốn trong comparison_runs[{current_problem}]: {type(run_data)} - {run_data}")
                continue # Bỏ qua mục này nếu nó không phải là dictionary
            
            history = run_data['history'] 
            label = run_data['label']
            self.ax_conv.plot(history, label=label, color=ccolors(i), linewidth=1.5)
            if len(history) > max_len:
                max_len = len(history)

        if max_len > 0: 
            self.ax_conv.set_xlim(left=0, right=max_len)
        self.ax_conv.legend(loc='best', fontsize='small')
        self.ax_conv.relim()
        self.ax_conv.autoscale_view()
        self.canvas.draw()

    def _update_saved_runs_listbox(self):
        """Cập nhật Listbox hiển thị các lần chạy đã lưu cho bài toán hiện tại."""
        self.saved_runs_listbox.delete(0, tk.END)
        current_problem = self.problem_var.get()
        
        if current_problem in self.comparison_runs:
            for i, run_data in enumerate(self.comparison_runs[current_problem]):
                # THÊM kiểm tra kiểu dữ liệu để tránh lỗi nếu có mục bị hỏng
                if isinstance(run_data, dict) and 'label' in run_data:
                    self.saved_runs_listbox.insert(tk.END, f"{i+1}. {run_data['label']}")
        
        self.delete_selected_button.config(state=tk.DISABLED) 

        # Kích hoạt nút "Hiển thị so sánh hội tụ" nếu có ít nhất 1 bản lưu (trước là 2)
        if len(self.comparison_runs.get(current_problem, [])) >= 1: # THAY ĐỔI TỪ 2 SANG 1
            self.compare_runs_button.config(state=tk.NORMAL)
        else:
            self.compare_runs_button.config(state=tk.DISABLED)

        self._update_side_by_side_comboboxes() # Cập nhật comboboxes chọn side-by-side
        # Không cần _clear_side_by_side_visualizations() ở đây nữa
        # vì _update_side_by_side_comboboxes() sẽ gọi _stop_parallel_visualization()


    def _check_parallel_animation_buttons(self):
        """Kiểm tra xem đã chọn đủ 2 run để kích hoạt nút chạy song song chưa."""
        if self.side_by_side_combo1.get() and self.side_by_side_combo2.get():
            self.run_parallel_animation_button.config(state=tk.NORMAL)
        else:
            self.run_parallel_animation_button.config(state=tk.DISABLED)
        
        # Dừng animation cũ ngay khi thay đổi lựa chọn combobox
        self._stop_parallel_visualization(reset_combos=False)
        self._clear_side_by_side_visualizations() # Xóa hiển thị cũ


    def _on_saved_run_selected(self, event):
        """Xử lý khi một bản lưu được chọn trong Listbox."""
        if self.saved_runs_listbox.curselection():
            self.delete_selected_button.config(state=tk.NORMAL)
        else:
            self.delete_selected_button.config(state=tk.DISABLED)

    def _delete_selected_comparison_run(self):
        selected_indices = self.saved_runs_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn một bản lưu để xóa.")
            return

        selected_listbox_index = selected_indices[0]
        current_problem = self.problem_var.get()
        
        if messagebox.askyesno("Xác nhận xóa", f"Bạn có chắc muốn xóa bản lưu số {selected_listbox_index + 1} không?"):
            if current_problem in self.comparison_runs and len(self.comparison_runs[current_problem]) > selected_listbox_index:
                del self.comparison_runs[current_problem][selected_listbox_index]
                messagebox.showinfo("Thông báo", "Đã xóa bản lưu thành công.")
            else:
                messagebox.showerror("Lỗi", "Không tìm thấy bản lưu để xóa.")
            
            self._update_saved_runs_listbox() # Cập nhật listbox sau khi xóa
            self.compare_all_saved_runs(clear_only=True) # Xóa đồ thị hội tụ
            self._stop_parallel_visualization() # Dừng animation song song nếu có
            self._clear_side_by_side_visualizations() # Xóa hiển thị side-by-side

    def _clear_all_comparisons(self):
        """Xóa tất cả các lần chạy đã lưu trong danh sách so sánh (cho TẤT CẢ các bài toán)."""
        if messagebox.askyesno("Xác nhận", "Bạn có chắc muốn xóa TẤT CẢ các lần chạy đã lưu để so sánh (cho TẤT CẢ bài toán) không?"):
            self.comparison_runs = {
                "Sphere Function": [],
                "Graph Coloring": []
            }
            self._update_saved_runs_listbox() # Cập nhật listbox sau khi xóa
            self.compare_all_saved_runs(clear_only=True) # Xóa đồ thị so sánh
            self._stop_parallel_visualization() # Dừng animation song song nếu có
            self._clear_side_by_side_visualizations() # Clear side-by-side plots
            messagebox.showinfo("Thông báo", "Đã xóa tất cả các lần so sánh.")
        else:
            return
        
    def _update_side_by_side_comboboxes(self):
        """Cập nhật các Combobox chọn bản lưu để hiển thị side-by-side."""
        current_problem = self.problem_var.get()
        run_labels = [run['label'] for run in self.comparison_runs.get(current_problem, [])]
        
        self.side_by_side_combo1['values'] = [''] + run_labels # Thêm lựa chọn trống
        self.side_by_side_combo2['values'] = [''] + run_labels
        
        self.side_by_side_combo1.set('') # Xóa lựa chọn hiện tại
        self.side_by_side_combo2.set('')
        self._clear_side_by_side_visualizations() # Xóa hiển thị cũ

    def _clear_side_by_side_visualizations(self):
        """Xóa nội dung và đặt lại tiêu đề của các trục so sánh song song."""
        self._stop_parallel_visualization(reset_combos=False) # Dừng animation trước

        self.ax_comp1.clear()
        self.ax_comp1.set_title("Kết quả Run 1")
        self.ax_comp1.set_xticks([])
        self.ax_comp1.set_yticks([])

        self.ax_comp2.clear()
        self.ax_comp2.set_title("Kết quả Run 2")
        self.ax_comp2.set_xticks([])
        self.ax_comp2.set_yticks([])
        self.canvas.draw_idle()

    def _update_side_by_side_visualizations(self):
        """Cập nhật nội dung của các trục so sánh song song dựa trên lựa chọn Combobox."""
        self._clear_side_by_side_visualizations()
        current_problem = self.problem_var.get()
        
        selected_label1 = self.side_by_side_combo1.get()
        selected_label2 = self.side_by_side_combo2.get()

        saved_runs_for_problem = self.comparison_runs.get(current_problem, [])
        
        run_data1 = next((run for run in saved_runs_for_problem if run['label'] == selected_label1), None)
        run_data2 = next((run for run in saved_runs_for_problem if run['label'] == selected_label2), None)

        if run_data1:
            self._display_saved_main_visualization(self.ax_comp1, run_data1)
        if run_data2:
            self._display_saved_main_visualization(self.ax_comp2, run_data2)
        
        self.canvas.draw_idle()


    def _display_saved_main_visualization(self, ax, run_data):
        """Vẽ trạng thái cuối cùng của một lần chạy đã lưu lên một trục cụ thể."""
        ax.clear()
        final_state = run_data['final_main_state']
        problem_name = self.problem_var.get() 
        
        ax.set_title(run_data['label'], fontsize=8) 
        ax.set_xticks([])
        ax.set_yticks([])

        if problem_name == "Sphere Function":
            # Sử dụng bounds đã lưu hoặc bounds hiện tại nếu không có
            bounds = final_state.get('bounds', (-5.12, 5.12)) 
            min_b, max_b = bounds[0], bounds[1]
            x = np.linspace(min_b, max_b, 100)
            y = np.linspace(min_b, max_b, 100)
            X, Y = np.meshgrid(x, y)
            Z = X**2 + Y**2 
            ax.contourf(X, Y, Z, levels=50, cmap='viridis')
            ax.scatter(final_state['positions'][:, 0], final_state['positions'][:, 1], c='red', s=10)
            ax.set_aspect('equal', adjustable='box') # Giữ tỷ lệ khung hình cho Sphere
        
        elif problem_name == "Graph Coloring":
            graph = final_state['graph']
            graph_pos = final_state['graph_pos']
            colors = final_state['colors']

            nx.draw_networkx_edges(graph, graph_pos, ax=ax)
            
            node_colors = []
            if colors is None or not (colors != -1).any(): 
                node_colors = ['grey'] * graph.number_of_nodes()
            else:
                assigned_colors = colors[colors != -1]
                unique_assigned_colors = np.unique(assigned_colors)
                
                if unique_assigned_colors.size == 0:
                    node_colors = ['grey'] * graph.number_of_nodes()
                else:
                    color_to_palette_index = {c: i for i, c in enumerate(unique_assigned_colors)}
                    cmap = plt.colormaps.get_cmap('viridis')
                    palette = cmap(np.linspace(0, 1, len(unique_assigned_colors)))
                    
                    for node_color_val in colors:
                        if node_color_val != -1 and node_color_val in color_to_palette_index:
                            node_colors.append(palette[color_to_palette_index[node_color_val]])
                        else:
                            node_colors.append('grey')
            
            nx.draw_networkx_nodes(graph, graph_pos, node_color=node_colors, node_size=300, ax=ax)
            ax.set_aspect('equal', adjustable='box') # Giữ tỷ lệ khung hình cho đồ thị


    def _start_parallel_visualization(self):
        # Dừng mọi animation song song đang chạy trước khi bắt đầu cái mới
        self._stop_parallel_visualization(reset_combos=False) 

        selected_label1 = self.side_by_side_combo1.get()
        selected_label2 = self.side_by_side_combo2.get()

        if not selected_label1 or not selected_label2:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn cả hai lần chạy để minh họa song song.")
            return

        current_problem = self.problem_var.get()
        runs_for_problem = self.comparison_runs.get(current_problem, [])

        run_data1 = next((run for run in runs_for_problem if run['label'] == selected_label1), None)
        run_data2 = next((run for run in runs_for_problem if run['label'] == selected_label2), None)

        if not run_data1 or not run_data2:
            messagebox.showerror("Lỗi", "Không tìm thấy dữ liệu cho một hoặc cả hai lần chạy đã chọn.")
            return

        # Chuẩn bị trục
        self.ax_comp1.clear()
        self.ax_comp1.set_title(run_data1['label'], fontsize=8)
        self.ax_comp1.set_xticks([])
        self.ax_comp1.set_yticks([])

        self.ax_comp2.clear()
        self.ax_comp2.set_title(run_data2['label'], fontsize=8)
        self.ax_comp2.set_xticks([])
        self.ax_comp2.set_yticks([])

        # Xác định số lượng frame tối thiểu
        num_frames = min(len(run_data1['animation_data']), len(run_data2['animation_data']))
        if num_frames == 0:
            messagebox.showwarning("Cảnh báo", "Không có dữ liệu animation cho các lần chạy đã chọn.")
            return

        # Cần lưu trữ các đối tượng plot để update
        self.plot_objects_comp1 = {}
        self.plot_objects_comp2 = {}
        # --- MỚI: Truyền dữ liệu frame đầu tiên để khởi tạo các đối tượng plot ---
        self._initialize_plot_objects(self.ax_comp1, self.plot_objects_comp1, run_data1['animation_data'][0], current_problem)
        self._initialize_plot_objects(self.ax_comp2, self.plot_objects_comp2, run_data2['animation_data'][0], current_problem)


        self.stop_parallel_animation_button.config(state=tk.NORMAL)
        self.run_parallel_animation_button.config(state=tk.DISABLED)

        # Tạo animation
        self.parallel_animation1 = FuncAnimation(
            self.fig, self._update_parallel_frame,
            fargs=(self.ax_comp1, self.plot_objects_comp1, run_data1['animation_data'], current_problem),
            frames=num_frames, interval=50, repeat=False, blit=False
        )
        self.parallel_animation2 = FuncAnimation(
            self.fig, self._update_parallel_frame,
            fargs=(self.ax_comp2, self.plot_objects_comp2, run_data2['animation_data'], current_problem),
            frames=num_frames, interval=50, repeat=False, blit=False
        )
        self.canvas.draw_idle()

    def _stop_parallel_visualization(self, reset_combos=True):
        """Dừng các animation song song và đặt lại trạng thái nút."""
        if self.parallel_animation1 and self.parallel_animation1.event_source:
            self.parallel_animation1.event_source.stop()
            self.parallel_animation1 = None
        if self.parallel_animation2 and self.parallel_animation2.event_source:
            self.parallel_animation2.event_source.stop()
            self.parallel_animation2 = None
        
        # Đặt lại trạng thái nút
        if hasattr(self, 'stop_parallel_animation_button'): # Kiểm tra để đảm bảo nút đã được tạo
            self.stop_parallel_animation_button.config(state=tk.DISABLED)
            if self.side_by_side_combo1.get() and self.side_by_side_combo2.get():
                 self.run_parallel_animation_button.config(state=tk.NORMAL)
            else:
                 self.run_parallel_animation_button.config(state=tk.DISABLED)

        if reset_combos: # Tùy chọn để không reset khi chỉ dừng animation
            if hasattr(self, 'side_by_side_combo1'): # Kiểm tra để đảm bảo comboboxes đã được tạo
                self.side_by_side_combo1.set('')
                self.side_by_side_combo2.set('')
            self._clear_side_by_side_visualizations() # Xóa visualization after stopping


    def _update_parallel_frame(self, frame_num, ax, plot_objects, animation_data, problem_name):
        """Hàm cập nhật cho mỗi frame của animation song song."""
        frame = animation_data[frame_num]

        if problem_name == "Sphere Function":
            positions = frame['positions']
            plot_objects['scatter'].set_offsets(positions)
            ax.set_title(f"{ax.get_title().split(' (')[0]} (Best: {frame['gbest_val']:.4e})", fontsize=8)

        elif problem_name == "Graph Coloring":
            colors = frame['colors']
            num_colors_display = frame['num_colors']
            num_conflicts_display = frame['num_conflicts']
            graph_nodes_collection = plot_objects['nodes_collection']

            if colors is not None:
                assigned_colors = colors[colors != -1]
                unique_assigned_colors = np.unique(assigned_colors)
                
                if unique_assigned_colors.size == 0:
                    node_colors = ['grey'] * len(colors)
                else:
                    # --- THÊM DÒNG NÀY ---
                    color_to_palette_index = {c: i for i, c in enumerate(unique_assigned_colors)}
                    cmap = plt.colormaps.get_cmap('viridis')
                    palette = cmap(np.linspace(0, 1, len(unique_assigned_colors)))
                    node_colors = [palette[color_to_palette_index[c]] if c != -1 and c in color_to_palette_index else 'grey' for c in colors]
                
                graph_nodes_collection.set_color(node_colors)
            else:
                graph_nodes_collection.set_color('grey')

            ax.set_title(f"{ax.get_title().split(' | ')[0]} | Màu: {num_colors_display} | Xung đột: {num_conflicts_display}", fontsize=8)

        return plot_objects.values()

    def _initialize_plot_objects(self, ax, plot_objects_dict, initial_frame_data, problem_name):
        """Khởi tạo các đối tượng plot ban đầu trên một trục và lưu chúng vào dict."""
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        
        if problem_name == "Sphere Function":
            bounds = initial_frame_data.get('bounds', (-5.12, 5.12)) 
            min_b, max_b = bounds[0], bounds[1]
            x = np.linspace(min_b, max_b, 100)
            y = np.linspace(min_b, max_b, 100)
            X, Y = np.meshgrid(x, y)
            Z = X**2 + Y**2 
            ax.contourf(X, Y, Z, levels=50, cmap='viridis')
            plot_objects_dict['scatter'] = ax.scatter(initial_frame_data['positions'][:, 0], initial_frame_data['positions'][:, 1], c='red', s=10)
            ax.set_aspect('equal', adjustable='box')

        elif problem_name == "Graph Coloring":
            graph = initial_frame_data['graph']
            graph_pos = initial_frame_data['graph_pos']
            colors = initial_frame_data['colors']

            nx.draw_networkx_edges(graph, graph_pos, ax=ax)
            
            node_colors = []
            if colors is None or not (colors != -1).any(): 
                node_colors = ['grey'] * graph.number_of_nodes()
            else:
                assigned_colors = colors[colors != -1]
                unique_assigned_colors = np.unique(assigned_colors)
                # --- THÊM DÒNG NÀY ---
                color_to_palette_index = {c: i for i, c in enumerate(unique_assigned_colors)}
                cmap = plt.colormaps.get_cmap('viridis')
                palette = cmap(np.linspace(0, 1, len(unique_assigned_colors)))
                node_colors = [palette[color_to_palette_index[c]] if c != -1 and c in color_to_palette_index else 'grey' for c in colors]
            
            plot_objects_dict['nodes_collection'] = nx.draw_networkx_nodes(graph, graph_pos, node_color=node_colors, node_size=300, ax=ax)
            # nx.draw_networkx_labels(graph, graph_pos, ax=ax, font_size=8, font_color='white')
            ax.set_aspect('equal', adjustable='box')
        
        return plot_objects_dict.values()
    
    def reset_current_visualization_state(self):
        """Dừng và đặt lại quá trình trực quan hóa hiện tại (ax_main và ax_conv)."""
        if self.animation and self.animation.event_source is not None:
            self.animation.event_source.stop()
        
        self.animation = None
        self.algorithm_runner = None
        self.current_run_animation_data = [] # Xóa dữ liệu animation của lần chạy hiện tại

        self.ax_main.clear()
        self.ax_conv.clear() 
        # Đảm bảo các trục được đặt lại tiêu đề và nhãn mặc định
        self.ax_main.set_title("Trực quan hóa Thuật toán")
        self.ax_main.set_xlabel("")
        self.ax_main.set_ylabel("")

        self.ax_conv.set_title("Lịch sử hội tụ (Best Fitness)")
        self.ax_conv.set_xlabel("Lần lặp")
        self.ax_conv.set_ylabel("Giá trị Fitness (log)")
        self.ax_conv.grid(True)
        self.ax_conv.set_yscale('log') # Luôn giữ log scale

        self.canvas.draw()

        self.start_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED, text="Tạm dừng")
        self.save_result_button.config(state=tk.DISABLED)
        

if __name__ == "__main__":
    root = tk.Tk()
    app = AlgorithmVisualizer(root)
    root.mainloop()