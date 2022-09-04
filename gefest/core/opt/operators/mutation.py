import copy
import random
from copy import deepcopy
from multiprocessing import Pool

import numpy as np

from gefest.core.algs.postproc.resolve_errors import postprocess
from gefest.core.opt.constraints import check_constraints
from gefest.tools.samplers.standard.standard import MAX_ITER, NUM_PROC, StandardSampler
from gefest.core.structure.domain import Domain
from gefest.core.structure.structure import Structure, get_random_poly, get_random_point
from gefest.core.structure.point import Point


def mutation(structure: Structure, domain: Domain, rate=0.6):
    """
    We divide mutations into two types: points mutations and polygons mutations
    Points mutation: add/delete points, change position
    Polygon mutation: add/delete polygon, rotate, resize
    """

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
            elif structure is None:
                # if the mutation did not return anything,
                # then it is considered unsuccessful,
                # in which case a random structure is generated
                new_structure = StandardSampler().get_pop_worker(domain=domain)
                is_correct = True
                break
    return new_structure


def polygons_mutation(new_structure: Structure, polygon_to_mutate_idx, domain: Domain):
    # Weights for each type of mutation
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
        new_poly = get_random_poly(parent_structure=new_structure,
                                   domain=domain)
        if new_poly is None:
            return new_structure
        new_structure.polygons.append(new_poly)
    elif random.random() < polygon_rotate_mutation_prob:
        # if rotate polygon
        angle = float(random.randint(-120, 120))
        geometry.rotate_poly(new_structure.polygons[polygon_to_mutate_idx], angle)
    elif random.random() < polygon_reshape_mutation_prob:
        # if resize polygon
        geometry.resize_poly(new_structure.polygons[polygon_to_mutate_idx],
                             x_scale=np.random.uniform(0.25, 3, 1)[0],
                             y_scale=np.random.uniform(0.25, 3, 1)[0])

    return new_structure


def add_delete_point_mutation(new_structure: Structure, polygon_to_mutate_idx, mutate_point_idx, domain):
    # Weight for add and delete point
    point_drop_mutation_prob = 0.5
    point_add_mutation_prob = 0.5

    # Choosing polygon and point to mutate
    polygon_to_mutate = new_structure.polygons[polygon_to_mutate_idx]
    point_to_mutate = polygon_to_mutate.points[mutate_point_idx]

    if (random.random() < point_drop_mutation_prob and
            len(polygon_to_mutate.points) > domain.min_points_num):
        # if drop point from polygon
        new_structure.polygons[polygon_to_mutate_idx].points.remove(point_to_mutate)
    else:
        # if add point to polygon
        new_point = get_random_point(polygon_to_mutate,
                                     new_structure,
                                     domain)

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
    # Neighborhood to reposition
    eps_x = round(domain.len_x / 10)
    eps_y = round(domain.len_y / 10)

    structure = copy.deepcopy(new_structure)

    # Displacement in the neighborhood
    displacement_x = random.randint(-eps_x, eps_x)
    displacement_y = random.randint(-eps_y, eps_y)

    x_new = structure.polygons[polygon_to_mutate_idx].points[mutate_point_idx].x + displacement_x
    y_new = structure.polygons[polygon_to_mutate_idx].points[mutate_point_idx].y + displacement_y

    i = 20  # Number of attempts to change the position of the point
    while not domain.contains(Point(x_new, y_new)):
        x_new = structure.polygons[polygon_to_mutate_idx].points[mutate_point_idx].x + displacement_x
        y_new = structure.polygons[polygon_to_mutate_idx].points[mutate_point_idx].y + displacement_y
        i -= 1
        if i == 0:
            # If number of attempts is over,
            # then transformation is unsuccessful
            # and returns input stucture
            return new_structure

    structure.polygons[polygon_to_mutate_idx].points[mutate_point_idx].x = x_new
    structure.polygons[polygon_to_mutate_idx].points[mutate_point_idx].y = y_new

    return structure


def points_mutation(new_structure: Structure, polygon_to_mutate_idx, domain: Domain):
    # Choosing type of points mutation, polygon to mutate and point to mutate

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
                    # If fixed points were removed from the structure after the mutation,
                    # they must be added back
                    if not (fixed.points == [p.points for p in new_structure.polygons]):
                        new_structure.polygons.append(deepcopy(fixed))

        new_structure = postprocess(new_structure, domain)
        constraints = check_constraints(structure=new_structure, domain=domain)
        max_attempts = 3  # Number of attempts to postprocess mutated structures
        while not constraints:
            new_structure = postprocess(new_structure, domain)
            constraints = check_constraints(structure=new_structure, domain=domain)
            max_attempts -= 1
            if max_attempts == 0:
                # If attempts is over,
                # mutation is considered like unsuccessful
                return None

        return new_structure
    except Exception as ex:
        print(f'Mutation error: {ex}')
        import traceback
        print(traceback.format_exc())
        return None
