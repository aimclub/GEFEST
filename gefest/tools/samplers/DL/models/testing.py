from gefest.tools.samplers.DL.microfluid_sampler import DeepSampler
from gefest.tools.samplers.standard.standard import StandardSampler
from gefest.core.structure.domain import Domain
import time

domain = Domain(allowed_area=[(-125, 100),
                              (-75, 170),
                              (15, 170),
                              (30, 90),
                              (-20, -130),
                              (-20, -170),
                              (-125, -170),
                              (-125, 100)],
                max_poly_num=7,
                min_poly_num=1,
                max_points_num=20,
                min_points_num=5,
                is_closed=True
                )
allowed_area = [(-125, 100),
                (-75, 170),
                (15, 170),
                (30, 90),
                (-20, -130),
                (-20, -170),
                (-125, -170),
                (-125, 100)]

c = [cs[0] for cs in allowed_area]
z = [cs[1] for cs in allowed_area]
import matplotlib.pyplot as plt

g = DeepSampler()
start = time.time()
tt = g.sample(1000, domain)
end = time.time()

standard_sampler = StandardSampler()
start = time.time()
tt = standard_sampler.sample(100, domain)
for i, t in enumerate(tt):
    plt.fill(c, z, color='black')
    for poly in t.polygons:
        x = [c.x for c in poly.points]
        y = [c.y for c in poly.points]
        plt.ylim(-170, 170)
        plt.xlim(-125, 30)
        plt.fill(x, y, color='white')
    plt.axis('off')
    plt.savefig(f'samples1/{i}.png', bbox_inches='tight', pad_inches=0)
    plt.close('all')

end = time.time()
