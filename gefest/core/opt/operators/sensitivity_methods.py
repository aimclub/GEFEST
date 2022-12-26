from copy import deepcopy
import random
from multiprocessing import Pool

from gefest.core.algs.postproc.resolve_errors import postprocess
from gefest.core.opt.constraints import check_constraints
from gefest.core.structure.domain import Domain
from gefest.core.structure.structure import Structure
from gefest.core.structure.polygon import Polygon
from gefest.core.structure.point import Point
from gefest.core.geometry.geometry_2d import Geometry2D
from gefest.core.structure.structure import get_structure_from_path
from cases.main_conf import opt_params

MAX_ITER = 50000
NUM_PROC = 1

geometry = Geometry2D()

def sa_mutation(structure: Structure, domain: Domain, rate=0.6) -> Structure:

    random_val = random.random()

    if random_val > rate:
        return structure

    is_correct = False

    changes_num = len(structure.polygons)

    n_iter = 0

    new_structure = deepcopy(structure)

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
                from gefest.tools.samplers.sens_analysis.sens_sampler import SensitivitySampler
                new_structure = SensitivitySampler(path=opt_params.structure_path).get_pop_worker(domain=domain)
                is_correct = True
                break
    return new_structure


def mutate_worker(args):
    structure, changes_num, domain = args[0], args[1], args[2]

    try:
        new_structure = deepcopy(structure)

        for _ in range(changes_num):
            mutations = [change_position, removing_point, rotate_poly]
            chosen_mutation = random.choice(mutations)
            polygon_to_mutate_idx = random.randint(0, len(new_structure.polygons) - 1)
            polygon_to_mutate = deepcopy(new_structure.polygons[polygon_to_mutate_idx])

            mutated_polugon = chosen_mutation(polygon_to_mutate)

            new_structure.polygons[polygon_to_mutate_idx] = mutated_polugon

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


def change_position(polygon: Polygon):
    moving_step = geometry.get_length(polygon=polygon)*0.05
    directions = ['north', 'south', 'east', 'west', 'n-w', 's-w', 'n-e', 's-e']

    chosen_direct = random.choice(directions)

    moved_poly = deepcopy(polygon)
    for idx, point in enumerate(moved_poly.points):
        moved_poly.points[idx] = moving_point(chosen_direct, point, moving_step)
    
    return moved_poly


def moving_point(direction: str, point: Point, moving_step) -> Point:
    directions = {'north': Point(point.x + moving_step, point.y),
                  'south': Point(point.x - moving_step, point.y),
                  'east': Point(point.x, point.y + moving_step),
                  'west': Point(point.x, point.y - moving_step),
                  'n-w': Point(point.x + moving_step, point.y - moving_step),
                  's-w': Point(point.x - moving_step, point.y + moving_step),
                  'n-e': Point(point.x + moving_step, point.y + moving_step),
                  's-e': Point(point.x - moving_step, point.y - moving_step)}
    return directions[direction]


def removing_point(polygon: Polygon):
    if len(polygon.points) > 2:
        polygon = deepcopy(polygon)
        points = polygon.points
        rand_idx = random.randint(0, len(points)-1)
        points.pop(rand_idx)
        polygon.points = points

        return polygon
    else:
        return polygon


def get_structure_for_analysis(path: str):
    structure = get_structure_from_path(path=path)
    if random.random() < 0.1:
        polygons = structure.polygons
        rand_idx = random.randint(0, len(polygons)-1)
        polygons.pop(rand_idx)
        structure.polygons = polygons

    methods = [change_position, removing_point, rotate_poly]
    for idx, polygon in enumerate(structure.polygons):
        chosen_method = random.choice(methods)
        new_poly = chosen_method(polygon)
        structure.polygons[idx] = new_poly

    return structure


def rotate_poly(polygon: Polygon):
    angles = [0, 90, 180, 270]
    poly = deepcopy(polygon)
    angle = random.choice(angles)
    rotated_poly = geometry.rotate_poly(poly=poly, angle=angle)

    return rotated_poly
    