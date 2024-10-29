import numpy as np
from itertools import product 
from utils.graph_controller import GraphController
from utils.target_controller import VTSPGaussian
import matplotlib.pyplot as plt
from phd_filter import PHDFilterWrapper
from arguments import arg

def add_t(X, t: float):
    return np.concatenate([X, np.full((X.shape[0], 1), t)], axis=1)

class PHDEnv:
    def __init__(self, graph_size, k_size, budget_size=None, target_size=None, start=None, obstacles=None):
        self.graph_size = graph_size
        self.k_size = k_size
        self.budget = self.budget_init = budget_size
        self.target_size = target_size
        self.obstacles = obstacles
        self.graph = GraphController(graph_size, k_size, start, obstacles)
        
        if start is None:
            self.start = np.random.rand(1, 2)
        else:
            self.start = np.array([start])

        self.curr_t = 0.0
        self.n_targets = target_size
        self. visit_t = [[] for _ in range(self.n_targets)]
        self.graph_ctrl = GraphController(self.graph_size, self.start, self.k_size, self.obstacles)
        self.node_coords, self.graph = self.graph_ctrl.generate_graph()

        # underlying distribution
        self.underlying_distrib = None
        self.ground_truth = None
        self.high_info_idx = None

        # PHD Filter
        self.phd_wrapper = None
        self.node_feature = None
        self.RMSE = None
        self.JS, self.JS_init, self.JS_list, self.KL, self.KL_init, self.KL_list = None, None, None, None, None, None
        self.cov_trace, self.cov_trace_init = None, None
        self.unc, self.unc_list, self.unc_init, self.unc_sum, self.unc_sum_list = None, None, None, None, None

        # start point
        self.current_node_index = 0
        self.dist_residual = 0
        self.sample = self.start.copy()
        self.random_speed_factor = None
        self.d_to_target = None
        self.route = []
        self.frame_files = []

    def reset(self, seed=None):
        # TODO: underlying distribution

        # TODO: initialize PHD Filter

        # TODO: initialize particles with prior knowledge
        self.phd_wrapper.init_particles()

        # TODO: initialize evaluations
        self.RMSE = self.eval_avg_RMSE(self.ground_truth)
        self.cov_trace = self.eval_avg_cov_trace()
        self.unc, self.unc_list = self.phd_wrapper.evaluate_unc()
        # TODO: other metrics

        return self.node_coords, self.graph, self.node_feature, self.budget
    
    def step(self, next_node_index, global_step=0, eval_speed=None):
        # TODO: update particles
        pass

    def get_ground_truth(self):
        x1 = np.linspace(0, 1, 40)
        x2 = np.linspace(0, 1, 40)
        x1x2 = np.array(list(product(x1, x2)))
        ground_truth = self.underlying_distrib.fn(x1x2)

        return ground_truth
    
    def get_high_info_idx(self):
        high_info_idx = []

        # TODO: need to add PHD Filter to get high info idx

        return high_info_idx
    
    def plot(self, route, n, step, path, budget_list, rew_list, div_list):
        # TODO: plot shortest path

        pass


if __name__ == '__main__':
    pass


