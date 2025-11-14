import numpy as np
import time
import matplotlib.pyplot as plt

class SphereFunction:
    """
    Bao đóng hàm Sphere và gradient của nó.
    """
    @staticmethod
    def objective(x):
        return np.sum(x ** 2, axis=-1)

    @staticmethod
    def gradient(x):
        return 2 * x


class GradientDescentHillClimbing:
    """
    Thuật toán Gradient Descent được đóng gói, giờ đây có thêm phương thức
    để tự vẽ biểu đồ hội tụ.
    """
    def __init__(self, objective_func=SphereFunction.objective, 
                 gradient_func=SphereFunction.gradient, 
                 learning_rate=0.1, 
                 max_steps=1000, 
                 search_range=(-5.12, 5.12),
                 population_size=50):
        self.objective_func = objective_func
        self.gradient_func = gradient_func
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.search_range = search_range
        self.population_size = population_size
        
        self.history = None    
        self.dimensions = None 

    def run(self, dimensions):
        """
        Sửa đổi: Giờ đây phương thức này sẽ lưu lại lịch sử hội tụ.
        """
        self.dimensions = dimensions
        lower_bound, upper_bound = self.search_range
        population = np.random.uniform(lower_bound, upper_bound, size=(self.population_size, dimensions))
        
        history_data = {'best': [], 'average': [], 'worst': []}

        for _ in range(self.max_steps):
            fitness_values = self.objective_func(population)
            history_data['best'].append(np.min(fitness_values))
            history_data['average'].append(np.mean(fitness_values))
            history_data['worst'].append(np.max(fitness_values))
            
            grad = self.gradient_func(population)
            population = population - self.learning_rate * grad

        self.history = history_data
        
        final_fitness_values = self.objective_func(population)
        best_idx = np.argmin(final_fitness_values)
        best_final_point = population[best_idx]
        best_final_fitness = final_fitness_values[best_idx]
        
        return best_final_point, best_final_fitness

    def visualize(self, img_path):
        """
        HÀM MỚI: Vẽ biểu đồ hội tụ dựa trên lịch sử đã lưu.
        Hàm này rất giống với mẫu bạn cung cấp.
        """
        if not self.history:
            print("Chưa có lịch sử để vẽ. Vui lòng chạy thuật toán bằng phương thức .run() trước.")
            return

        plt.figure(figsize=(12, 7))
        
        plt.plot(self.history['best'], color='blue', linestyle='-', label='Best')
        plt.plot(self.history['average'], color='green', linestyle='--', label='Average')
        plt.plot(self.history['worst'], color='red', linestyle=':', label='Worst')
        
        plt.yscale('log')
        
        plt.title(f"Hill Climbing Convergence Curve - Dim={self.dimensions}", fontsize=16)
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Fitness f(x)", fontsize=12)
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(img_path, dpi=300)
        print(f"Đã lưu biểu đồ vào: {img_path}")
        plt.show()

 