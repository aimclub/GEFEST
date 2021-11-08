from gefest.core.algs.geom.validation import out_of_bound, self_intersection, too_close
from gefest.core.algs.postproc.resolve_errors import postprocess
from gefest.core.structure.structure import Structure


def check_constraints(structure: Structure, is_lightweight: bool = False, domain=None, model_func=None) -> bool:
    try:
        if any([(poly is None or
                 len(poly.points) == 0 or
                 any([pt is None for pt in poly.points]))
                for poly in structure.polygons]):
            print('Wrong structure - problems with points')
            return False

        # final postprocessing
        structure = postprocess(structure, domain)

        cts = [out_of_bound(structure, domain),
               too_close(structure, domain),
               self_intersection(structure)]
        structurally_correct = not any(cts)

        if structurally_correct and not is_lightweight:
            print('Check heavy constraint')
            obj, _, _ = model_func(structure)
            return -obj < 0

        if not structurally_correct:
            print(f'Constraint violated in {domain.name}: {cts}')
            structure.plot(domain,
                           title=f'Constraint violated in {domain.name}: {cts}')
            return False
    except Exception as ex:
        print(ex)
        import traceback
        print(traceback.format_exc())
        return False

    return structurally_correct
