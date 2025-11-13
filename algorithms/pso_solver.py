import numpy as np

class ParticleSwarmOptimizationSphere:
    def __init__(self, objective_func, bounds, n_particles, n_iter, c1, c2, w):
        self.func = objective_func
        self.bounds = bounds
        self.n_particles = n_particles
        self.n_dims = len(bounds[0]) if hasattr(bounds[0], '__len__') else 2 # Default to 2D
        
        self.c1 = c1
        self.c2 = c2
        self.w = w
        
        min_bound, max_bound = self.bounds
        self.positions = np.random.uniform(min_bound, max_bound, (n_particles, self.n_dims))
        self.velocities = np.zeros((n_particles, self.n_dims))
        
        self.pbest_pos = self.positions.copy()
        self.pbest_val = np.array([self.func(p) for p in self.positions])
        
        gbest_idx = np.argmin(self.pbest_val)
        self.gbest_pos = self.pbest_pos[gbest_idx]
        self.gbest_val = self.pbest_val[gbest_idx]
        
        self.history = [self.gbest_val]

    def step(self):
        r1 = np.random.rand(self.n_particles, self.n_dims)
        r2 = np.random.rand(self.n_particles, self.n_dims)

        cognitive_vel = self.c1 * r1 * (self.pbest_pos - self.positions)
        social_vel = self.c2 * r2 * (self.gbest_pos - self.positions)
        self.velocities = self.w * self.velocities + cognitive_vel + social_vel

        self.positions += self.velocities
        min_bound, max_bound = self.bounds
        self.positions = np.clip(self.positions, min_bound, max_bound)

        current_val = np.array([self.func(p) for p in self.positions])
        
        update_mask = current_val < self.pbest_val
        self.pbest_pos[update_mask] = self.positions[update_mask]
        self.pbest_val[update_mask] = current_val[update_mask]

        current_gbest_idx = np.argmin(self.pbest_val)
        if self.pbest_val[current_gbest_idx] < self.gbest_val:
            self.gbest_pos = self.pbest_pos[current_gbest_idx]
            self.gbest_val = self.pbest_val[current_gbest_idx]
        
        self.history.append(self.gbest_val)
        
        return {
            "positions": self.positions,
            "gbest_val": self.gbest_val,
            "history": self.history
        }
    
    def _get_state(self): 
        return {
            "positions": self.positions,
            "gbest_val": self.gbest_val,
            "history": self.history
        }