import numpy as np
import time

class SphereFunction:
    """
    Bao đóng hàm Sphere và gradient của nó.
    """
    @staticmethod
    def objective(x):
        return np.sum(x ** 2)

    @staticmethod
    def gradient(x):
        return 2 * x


class GradientDescentHillClimbing:
    """
    Thuật toán Gradient Descent được đóng gói thành class.
    Logic hoàn toàn giống bản hàm ban đầu.
    """
    def __init__(self, objective_func=SphereFunction.objective, gradient_func=SphereFunction.gradient, learning_rate=0.1, max_steps=1000, search_range=(-5.12, 5.12)):
        self.objective_func = objective_func
        self.gradient_func = gradient_func
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.search_range = search_range

    def run(self, dimensions):
        lower_bound, upper_bound = self.search_range

        # Khởi tạo điểm ngẫu nhiên
        current_point = np.random.uniform(lower_bound, upper_bound, size=dimensions)

        for _ in range(self.max_steps):
            grad = self.gradient_func(current_point)
            new_point = current_point - self.learning_rate * grad

            if np.allclose(current_point, new_point):
                break

            current_point = new_point

        final_value = self.objective_func(current_point)
        return current_point, final_value

