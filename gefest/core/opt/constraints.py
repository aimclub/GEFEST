import copy

from gefest.core.algs.geom.validation import out_of_bound, self_intersection, too_close, intersection, unclosed_poly
from gefest.core.structure.structure import Structure


def check_constraints(structure: Structure, is_lightweight: bool = False, domain=None, model_func=None):
    keys = [
        'out_of_bound',
        'too_close',
        'self_intersection',
        'intersection',
        'unclosed_poly',
    ]
    try:
        if any([(poly is None or
                 len(poly.points) == 0 or
                 any([pt is None for pt in poly.points]))
                for poly in structure.polygons]):
            print('Wrong structure - problems with points')
            return False

        cts = [out_of_bound(structure, domain),
               too_close(structure, domain),
               self_intersection(structure),
               intersection(structure, domain.geometry),
               unclosed_poly(structure, domain)]
        structurally_correct = not any(cts)
        constraints = dict(zip(keys, cts))

        if structurally_correct:
            return None
    except Exception as ex:
        print(ex)
        import traceback
        print(traceback.format_exc())
        return False

    return constraints
