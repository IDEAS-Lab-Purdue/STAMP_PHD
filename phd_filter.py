from itertools import product
import numpy as np
import matplotlib.pyplot as plt

def add_t(X, t: float):
    return np.concatenate([X, np.full((X.shape[0], 1), t)], axis=1)

class PHDFilter:
    def __init__(self, node_coords):
        self.node_coords = node_coords
        self.grid_size = 40
        self.grid = np.array(list(product(np.linspace(0, 1, self.grid_size), np.linspace(0, 1, self.grid_size))))
        self.curr_t = None

        # PHD Filter params
        self.particles = [] # each particle a 'state' and 'weight' dict
        self.birth_intensity = 0.05
        self.survival_probability = 0.99
        self.detection_probability = 0.9
        self.clutter_intensity = 1e-5
        self.motion_cov = np.diag([0.01, 0.01])
        self.observation_cov = np.diag([0.01, 0.01])

    def init_particles(self, n_particles=1000):
        # random initialization uniformly over grid
        indices = np.random.choice(len(self.grid), n_particles)
        positions = self.grid[indices]
        weights = np.full(n_particles, 1.0/n_particles)
        self.particles = [{'state': pos, 'weight': w} for pos, w in zip(positions, weights)]

    def predict(self):
        for particle in self.particles:
            noise = np.random.multivariate_normal([0, 0], self.motion_cov)
            particle['state'] += noise
            particle['state'] = np.clip(particle['state'], 0, 1) # bound
            particle['weight'] *= self.survival_probability

        # birth
        n_birth_particles = int(self.birth_intensity * len(self.particles))
        indices = np.random.choice(len(self.grid), n_birth_particles)
        birth_positions = self.grid[indices]
        birth_weights = np.full(n_birth_particles, self.birth_intensity)
        # birth weights = np.full(n_birth_particles, 1.0/n_birth_particles)
        birth_particles = [{'state': pos, 'weight': w} for pos, w in zip(birth_positions, birth_weights)]
        self.particles.extend(birth_particles)

    def update(self, observations):
        for particle in self.particles:
            likelihood = self.clutter_intensity
            for obs in observations:
                diff = particle['state'] - obs
                exponent = -0.5 * diff.T @ np.linalg.inv(self.observation_cov) @ diff
                det_cov = np.linalg.det(2 * np.pi * self.observation_cov)
                likelihood += self.detection_probability * np.exp(exponent) / np.sqrt(det_cov)

            particle['weight'] *= likelihood

        # normalize weights
        total_weight = sum(p['weight'] for p in self.particles)
        if total_weight > 0:
            for particle in self.particles:
                particle['weight'] /= total_weight

        weight_threshold = 1e-5
        self.particles = [p for p in self.particles if p['weight'] > weight_threshold]

    def estimate(self):
        if not self.particles:
            return None
        positions = np.array([p['state'] for p in self.particles])
        weights = np.array([p['weight'] for p in self.particles])
        estimated_pos = np.average(positions, axis=0, weights=weights)
        return estimated_pos
    
    def evaluate_uncertainty(self):

        pass

    def evaluate_JS_divergence(self):

        pass

    def plot(self, true_pos=None, agent_pos=None):

        pass


class PHDFilterWrapper:
    def __init__(self, num_targets, node_coords):
        self.num_targets = num_targets
        self.node_coords = node_coords
        self.phd_filters = [PHDFilter(node_coords) for _ in range(num_targets)]

        for phd_filter in self.phd_filters:
            phd_filter.init_particles()

    def predict(self):
        for phd in self.phd_filters:
            phd.predict()

    def update(self, observations):
        for phd, obs in zip(self.phd_filters, observations):
            phd.update(obs)

    def estimate(self):
        estimates = []
        for phd in self.phd_filters:
            estimates.append(phd.estimate())
        return estimates
    
    def evaluate_uncertainty(self):
        uncertainties = []
        for phd in self.phd_filters:
            uncertainties.append(phd.evaluate_uncertainty())
        avg_uncertainty = np.mean(uncertainties)
        return avg_uncertainty, uncertainties
    
    def plot(self, true_pos=None, agent_pos=None):
        # fig, ax = plt.subplots()
        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 1)
        # ax.set_aspect('equal')
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_title('PHD Filter')

        # if true_pos is not None:
        #     ax.scatter(true_pos[0], true_pos[1], c='r', marker='x', label='True Position')
        # if agent_pos is not None:
        #     ax.scatter(agent_pos[0], agent_pos[1], c='b', marker='x', label='Agent Position')

        # for phd in self.phd_filters:
        #     positions = np.array([p['state'] for p in phd.particles])
        #     ax.scatter(positions[:, 0], positions[:, 1], s=1, alpha=0.5)

        # plt.legend()
        # plt.show()

        pass
            