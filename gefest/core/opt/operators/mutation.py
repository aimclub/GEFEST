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


def mutation(structure: Structure, domain: Domain, rate=0.6):
    random_val = random.random()

    if random_val > rate:
        return structure

    is_correct = False

    changes_num = len(structure.polygons)

    n_iter = 0

    new_structure = structure

    while not is_correct and n_iter < MAX_ITER:
        n_iter += 1

        if NUM_PROC > 1:
            with Pool(NUM_PROC) as p:
                new_items = \
                    p.map(mutate_worker,
                          [[new_structure, changes_num, domain] for _ in range(NUM_PROC)])
        else:
            new_items = [mutate_worker([new_structure, changes_num, domain]) for _ in range(NUM_PROC)]

        for structure in new_items:
            if structure is not None:
                new_structure = structure
                is_correct = True
                break
    return new_structure


def polygons_mutation(new_structure: Structure, polygon_to_mutate_idx, domain: Domain):
    polygon_drop_mutation_prob = 0.2
    polygon_add_mutation_prob = 0.2
    polygon_rotate_mutation_prob = 0.5
    polygon_reshape_mutation_prob = 0.5

    geometry = domain.geometry

    if random.random() < polygon_drop_mutation_prob and len(new_structure.polygons) > 1:
        # if drop polygon from structure
        polygon_to_remove = new_structure.polygons[polygon_to_mutate_idx]
        new_structure.polygons.remove(polygon_to_remove)
    elif random.random() < polygon_add_mutation_prob and \
            len(new_structure.polygons) < domain.max_poly_num:
        # if add polygon to structure
        new_poly = get_random_poly(is_large=False,
                                   parent_structure=new_structure,
                                   domain=domain)
        if new_poly is None:
            return None
        new_structure.polygons.append(new_poly)
    elif random.random() < polygon_rotate_mutation_prob:
        # if add polygon to structure
        angle = float(random.randint(-60, 60))
        geometry.rotate_poly(new_structure.polygons[polygon_to_mutate_idx], angle)
    elif random.random() < polygon_reshape_mutation_prob:
        # if add polygon to structure
        geometry.resize_poly(new_structure.polygons[polygon_to_mutate_idx],
                             x_scale=np.random.uniform(0.25, 3, 1)[0],
                             y_scale=np.random.uniform(0.25, 3, 1)[0])

    return new_structure


def add_delete_point_mutation(new_structure: Structure, polygon_to_mutate_idx, mutate_point_idx, domain):
    point_drop_mutation_prob = 0.5
    point_add_mutation_prob = 0.5

    polygon_to_mutate = new_structure.polygons[polygon_to_mutate_idx]
    point_to_mutate = polygon_to_mutate.points[mutate_point_idx]

    if (random.random() < point_drop_mutation_prob and
            len(polygon_to_mutate.points) > domain.min_points_num):
        # if drop point from polygon
        new_structure.polygons[polygon_to_mutate_idx].points.remove(point_to_mutate)
    else:
        # if change point in polygon
        if point_to_mutate is not None and not domain.contains(point_to_mutate):
            raise ValueError('Wrong prev_point')

        new_point = get_random_point(point_to_mutate, polygon_to_mutate,
                                     new_structure, domain=domain)

        if new_point is None:
            return None

        if random.random() < point_add_mutation_prob:
            if mutate_point_idx + 1 < len(polygon_to_mutate.points):
                new_structure.polygons[polygon_to_mutate_idx].points.insert(mutate_point_idx + 1, new_point)
            else:
                new_structure.polygons[polygon_to_mutate_idx].points.insert(mutate_point_idx - 1, new_point)
        else:
            new_structure.polygons[polygon_to_mutate_idx].points[mutate_point_idx] = new_point

    return new_structure


def pos_change_point_mutation(new_structure: Structure, polygon_to_mutate_idx, mutate_point_idx, domain):
    eps_x = round(domain.len_x / 10)
    eps_y = round(domain.len_y / 10)

    displacement_x = random.randint(-eps_x, eps_x)
    displacement_y = random.randint(-eps_y, eps_y)

    x_old = new_structure.polygons[polygon_to_mutate_idx].points[mutate_point_idx].x
    y_old = new_structure.polygons[polygon_to_mutate_idx].points[mutate_point_idx].y

    new_structure.polygons[polygon_to_mutate_idx].points[mutate_point_idx].x = x_old + displacement_x
    new_structure.polygons[polygon_to_mutate_idx].points[mutate_point_idx].y = y_old + displacement_y

    return new_structure


def points_mutation(new_structure: Structure, polygon_to_mutate_idx, domain: Domain):
    polygon_to_mutate = new_structure.polygons[polygon_to_mutate_idx]

    mutate_point_idx = random.randint(0, len(polygon_to_mutate.points) - 1)
    point_to_mutate = polygon_to_mutate.points[mutate_point_idx]
    if point_to_mutate in domain.fixed_points:
        return None

    case = random.randint(0, 1)
    if case == 0:
        new_structure = add_delete_point_mutation(new_structure, polygon_to_mutate_idx, mutate_point_idx, domain)
    else:
        new_structure = pos_change_point_mutation(new_structure, polygon_to_mutate_idx, mutate_point_idx, domain)

    return new_structure


def mutate_worker(args):
    structure, changes_num, domain = args[0], args[1], args[2]
    polygon_mutation_probab = 0.5

    try:
        new_structure = copy.deepcopy(structure)

        for _ in range(changes_num):
            polygon_to_mutate_idx = random.randint(0, len(new_structure.polygons) - 1)
            case = random.random()

            if case < polygon_mutation_probab:
                new_structure = polygons_mutation(new_structure, polygon_to_mutate_idx, domain)
            else:
                new_structure = points_mutation(new_structure, polygon_to_mutate_idx, domain)

            if new_structure is None:
                continue

            for fixed in domain.fixed_points:
                if fixed not in new_structure.polygons:
                    new_structure.polygons.append(deepcopy(fixed))

        new_structure = postprocess(new_structure, domain)
        is_correct = check_constraints(new_structure, is_lightweight=True, domain=domain)
        if not is_correct:
            return None
        else:
            return new_structure
    except Exception as ex:
        print(f'Mutation error: {ex}')
        import traceback
        print(traceback.format_exc())
        return None
