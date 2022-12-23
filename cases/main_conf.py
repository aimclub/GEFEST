import argparse

"""
General configurations for all cases.
Some specific ones should be described in corresponding file
"""
parser = argparse.ArgumentParser()
parser.add_argument("--pop_size", type=int, default=50, help='number of individs in population')
parser.add_argument("--n_steps", type=int, default=200, help='number of generative design steps')
parser.add_argument('--n_polys', type=int, default=1, help='maximum number of polygons in structure')
parser.add_argument('--n_points', type=int, default=50, help='maximum number of points in polygon')
parser.add_argument('--c_rate', type=float, default=0.1, help='crossover rate')
parser.add_argument('--m_rate', type=float, default=0.9, help='mutation rate')
parser.add_argument('--is_closed', type=bool, default=True, help='type of polygon')
opt_params = parser.parse_args()
