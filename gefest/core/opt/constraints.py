from gefest.core.algs.postproc import postprocess
from gefest.core.structure.geometry import out_of_bound, self_intersection, too_close
from gefest.core.structure.structure import Structure
from gefest.core.utils import GlobalEnv


def check_constraints(structure: Structure, is_lightweight: bool = False, domain=None) -> bool:
    if domain is None:
        current_domain = GlobalEnv().domain
    else:
        current_domain = domain
    try:
        if any([(poly is None or
                 len(poly.points) == 0 or
                 any([pt is None for pt in poly.points]))
                for poly in structure.polygons]):
            print('Wrong structure - problems with points')
            return False

        # final postprocessing
        structure = postprocess(structure, current_domain)

        cts = [out_of_bound(structure, current_domain),
               too_close(structure, current_domain),
               self_intersection(structure)]
        structurally_correct = not any(cts)

        if structurally_correct and not is_lightweight:
            print('Check heavy constraint')
            model_func = GlobalEnv().model_func
            obj, _, _ = model_func(structure)
            return -obj < 0

        if not structurally_correct:
            print(f'Constraint violated in {current_domain.name}: {cts}')
            structure.plot(current_domain,
                           title=f'Constraint violated in {current_domain.name}: {cts}')
            return False
    except Exception as ex:
        print(ex)
        import traceback
        print(traceback.format_exc())
        return False

    return structurally_correct
