import copy
import random
from copy import deepcopy
from multiprocessing import Pool

import numpy as np

from gefest.core.algs.postproc.resolve_errors import postprocess
from gefest.core.opt.constraints import check_constraints
from gefest.core.opt.operators.initial import MAX_ITER, NUM_PROC, get_pop_worker
from gefest.core.structure.domain import Domain
from gefest.core.structure.polygon import Polygon
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

    n_iter = 0

    new_structure = deepcopy(structure)

    while not is_correct and n_iter < MAX_ITER:
        n_iter += 1

        if NUM_PROC > 1:
            with Pool(NUM_PROC) as p:
                new_items = \
                    p.map(mutate_worker,
                          [[new_structure, domain] for _ in range(NUM_PROC)])
        else:
            new_items = [mutate_worker(new_structure, domain) for _ in range(NUM_PROC)]

        for structure in new_items:
            if structure is not None:
                new_structure = structure
                is_correct = True
                break
            elif structure is None:
                # if the mutation did not return anything,
                # then it is considered unsuccessful,
                # in which case a random structure is generated
                new_structure = get_pop_worker(domain=domain)
                is_correct = True
                break
    return new_structure


def mutate_worker(structure: Structure, domain: Domain):

    # polygon_mutation_probab = 0.5
    mutation_ways = [polygons_mutation, points_mutation]
    
    try:
        new_polygons = []

        for polygon_to_mutate in structure.polygons:

            choosen_mutation_way = random.choice(mutation_ways)
            mutated_poly = choosen_mutation_way(polygon_to_mutate, domain)
            restriction = any([mutated_poly is None,
                               len(new_polygons) > domain.max_poly_num])
            if not restriction:
                new_polygons.append(mutated_poly)
            

        new_structure = Structure(new_polygons)
        if new_structure is None:
            return None

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


def polygons_mutation(poly_to_mutate: Polygon, domain: Domain):

    mutation_way = [drop_poly, add_poly, rotate_poly, resize_poly]
    choosen_way = random.choice(mutation_way)
    new_poly = choosen_way(poly_to_mutate, domain)

    return new_poly


def points_mutation(poly_to_mutate: Polygon, domain: Domain):

    mutate_point_idx = random.randint(0, len(poly_to_mutate.points) - 1)
    point_to_mutate = poly_to_mutate.points[mutate_point_idx]
    if point_to_mutate in domain.fixed_points:
        return poly_to_mutate
    
    mutation_way = [remove_point, add_point, change_position_point]
    choosen_way = random.choice(mutation_way)
    new_poly = choosen_way(poly_to_mutate, mutate_point_idx, domain)

    return new_poly


def remove_point(poly_to_mutate: Polygon, mutate_point_idx, domain: Domain):

    new_poly = deepcopy(poly_to_mutate)
    removing_point = poly_to_mutate.points[mutate_point_idx]
    new_poly.points.remove(removing_point)
    if len(new_poly.points) < domain.min_points_num:
        return poly_to_mutate
    return new_poly


def add_point(poly_to_mutate: Polygon, mutate_point_idx, domain: Domain):

    new_poly = deepcopy(poly_to_mutate)
    new_point = get_random_point(poly_to_mutate,
                                 Structure([poly_to_mutate]),
                                 domain)
    new_poly.points.insert(mutate_point_idx + 1, new_point)
    restriction = any([new_point is None,
                       len(new_poly.points) > domain.max_points_num])
    if restriction:
        return poly_to_mutate
    return new_poly


def change_position_point(poly_to_mutate: Polygon, mutate_point_idx, domain: Domain):

    new_poly = deepcopy(poly_to_mutate)
    # Neighborhood to reposition
    eps_x = round(domain.len_x / 10)
    eps_y = round(domain.len_y / 10)

    # Displacement in the neighborhood
    displacement_x = random.randint(-eps_x, eps_x)
    displacement_y = random.randint(-eps_y, eps_y)

    x_new = poly_to_mutate.points[mutate_point_idx].x + displacement_x
    y_new = poly_to_mutate.points[mutate_point_idx].y + displacement_y

    i = 20  # Number of attempts to change the position of the point
    while not domain.contains(Point(x_new, y_new)):
        x_new = poly_to_mutate.points[mutate_point_idx].x + displacement_x
        y_new = poly_to_mutate.points[mutate_point_idx].y + displacement_y
        i -= 1
        if i == 0:
            # If number of attempts is over,
            # then transformation is unsuccessful
            # and returns input stucture
            return poly_to_mutate

    new_poly.points[mutate_point_idx].x = x_new
    new_poly.points[mutate_point_idx].y = y_new

    return new_poly


def drop_poly(poly_to_mutate: Polygon, domain: Domain):

    return None


def add_poly(poly_to_mutate: Polygon, domain: Domain):
    structure = Structure(polygons=[])
    new_poly = get_random_poly(structure, domain)
    return new_poly


def rotate_poly(poly_to_mutate: Polygon, domain: Domain):

    angle = float(random.randint(-120, 120))
    new_poly = domain.geometry.rotate_poly(poly_to_mutate, angle)
    return new_poly


def resize_poly(poly_to_mutate: Polygon, domain: Domain):

    new_poly = domain.geometry.resize_poly(poly_to_mutate,
                                           x_scale=np.random.uniform(0.25, 3, 1)[0],
                                           y_scale=np.random.uniform(0.25, 3, 1)[0])
    return new_poly
