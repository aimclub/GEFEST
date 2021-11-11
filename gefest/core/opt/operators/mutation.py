import copy
import random
from copy import deepcopy
from multiprocessing import Pool

import numpy as np

from gefest.core.algs.postproc.resolve_errors import postprocess
from gefest.core.opt.constraints import check_constraints
from gefest.core.opt.operators.initial import MAX_ITER, NUM_PROC
from gefest.core.structure.domain import Domain
from gefest.core.structure.structure import Structure, get_random_point, get_random_poly


def mutation(structure: Structure,  domain: Domain, rate=0.6):
    random_val = random.random()

    if random_val > rate:
        return structure

    is_correct = False

    min_pol_size = 90
    changes_num = 1

    n_iter = 0

    new_structure = structure

    while not is_correct and n_iter < MAX_ITER:
        n_iter += 1
        print('mut', n_iter)

        if NUM_PROC > 1:
            with Pool(NUM_PROC) as p:
                new_items = \
                    p.map(mutate_worker,
                          [[new_structure, changes_num, min_pol_size, domain] for _ in range(NUM_PROC)])
        else:
            new_items = [mutate_worker([new_structure, changes_num, min_pol_size, domain]) for _ in range(NUM_PROC)]

        for structure in new_items:
            if structure is not None:
                #       is_correct = check_constraints(structure, domain=domain, is_lightweight=True)
                #       if is_correct:
                new_structure = structure
                is_correct = True
                break

    return new_structure


def mutate_worker(args):
    structure, changes_num, min_pol_size, domain = args[0], args[1], args[2], args[3]

    polygon_drop_mutation_prob = 0.2
    polygon_add_mutation_prob = 0.2
    point_drop_mutation_prob = 0.5
    point_add_mutation_prob = 0.2
    polygon_rotate_mutation_prob = 0.5
    polygon_reshape_mutation_prob = 0.5

    geometry = domain.geometry

    try:
        new_structure = copy.deepcopy(structure)

        for _ in range(changes_num):
            polygon_to_mutate = new_structure.polygons[random.randint(0, len(new_structure.polygons) - 1)]

            if random.random() < polygon_drop_mutation_prob and len(new_structure.polygons) > 1:
                # if drop polygon from structure
                new_structure.polygons.remove(polygon_to_mutate)
            elif random.random() < polygon_add_mutation_prob and \
                    len(new_structure.polygons) < domain.max_poly_num:
                # if add polygon to structure
                new_poly = get_random_poly(min_pol_size=90,
                                           max_pol_size=100,
                                           is_large=False,
                                           parent_structure=new_structure,
                                           domain=domain)
                if new_poly is None:
                    continue
                new_structure.polygons.append(new_poly)
            elif random.random() < polygon_rotate_mutation_prob:
                # if add polygon to structure
                angle = float(random.randint(-80, 80))
                polygon_to_mutate = geometry.rotate_poly(polygon_to_mutate, angle)
            elif random.random() < polygon_reshape_mutation_prob:
                # if add polygon to structure
                polygon_to_mutate = \
                    geometry.resize_poly(polygon_to_mutate,
                                         x_scale=max(0.25,
                                                     float(np.random.normal(1, 0.5, 1)[0])),
                                         y_scale=max(0.25,
                                                     float(np.random.normal(1, 0.5, 1)[0])))
            else:
                mutate_point_ind = random.randint(0, len(polygon_to_mutate.points) - 1)
                point_to_mutate = polygon_to_mutate.points[mutate_point_ind]
                if point_to_mutate in domain.fixed_points:
                    continue
                if (random.random() < point_drop_mutation_prob and
                        len(polygon_to_mutate.points) > min_pol_size):
                    # if drop point from polygon
                    polygon_to_mutate.points.remove(point_to_mutate)
                else:
                    # if change point in polygon

                    if point_to_mutate is not None and not domain.contains(point_to_mutate):
                        print("!!!!!!!!!!!!!!1")
                        raise ValueError('Wrong prev_point')

                    new_point = get_random_point(point_to_mutate, polygon_to_mutate,
                                                 new_structure, domain=domain)
                    if new_point is None:
                        continue

                    if random.random() < point_add_mutation_prob:
                        if mutate_point_ind + 1 < len(polygon_to_mutate.points):
                            polygon_to_mutate.points.insert(mutate_point_ind + 1, new_point)
                        else:
                            polygon_to_mutate.points.insert(mutate_point_ind - 1, new_point)
                    else:
                        polygon_to_mutate.points[mutate_point_ind] = new_point

            for fixed in domain.fixed_points:
                if fixed not in polygon_to_mutate.points:
                    polygon_to_mutate.points.append(deepcopy(fixed))

        new_structure = postprocess(new_structure, domain)
        is_correct = check_constraints(new_structure, is_lightweight=True, domain=domain)
        if not is_correct:
            return None
        return new_structure
    except Exception as ex:
        print(f'Mutation error: {ex}')
        import traceback
        print(traceback.format_exc())
        return None
