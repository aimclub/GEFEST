import gc
import os
import pickledb
import numpy as np
import mph
import pickle

from uuid import uuid4
from gefest.core.structure.structure import Structure

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
USE_AVG_CONST = False


class Comsol:
    """
    ::TODO:: make abstract class for further specific realizations
    """
    """
    Comsol class for microfluidic problem
    """

    def __init__(self, path_to_mph):
        """
        :param path_to_mph: (String), path to mph file
        """
        super(Comsol, self).__init__()

        self.client = mph.Client()
        self.path_to_mph = path_to_mph

    def estimate(self, structure: Structure):
        """
        Estimation using comsol multiphysics
        :param structure: (Structure), Structure of polygons
        :return: (Int), Performance
        """
        gc.collect()
        target, idx = self._load_fitness(structure)
        if target is None:
            model, idx = self._load_simulation_result(self.client, structure)
            if model is None:
                poly_box = []
                print('Start COMSOL')
                for i, pol in enumerate(structure.polygons):
                    poly_repr = []
                    poly_repr.append(' '.join([str(pt.x) for pt in pol.points]))
                    poly_repr.append(' '.join([str(pt.y) for pt in pol.points]))
                    poly_box.append(poly_repr)

                model = self.client.load(self.path_to_mph)

                try:
                    model = self._poly_add(model, poly_box)

                    model.build()
                    model.mesh()
                    model.solve()
                except Exception as ex:
                    print(ex)
                    self.client.clear()
                    return 0.0

                idx = self._save_simulation_result(structure, model)

            try:
                outs = [model.evaluate('vlct_1'),
                        model.evaluate('vlct_2'),
                        model.evaluate('vlct_3'),
                        model.evaluate('vlct_4'),
                        model.evaluate('vlct_5'),
                        model.evaluate('vlct_side'),
                        model.evaluate('vlct_main')]
            except Exception as ex:
                print(ex)
                self.client.clear()
                return 0.0

            u = model.evaluate('spf.U')
            curl = model.evaluate('curl')
            curv = model.evaluate('curv') / 10 ** 7

            fast_u_tresh = 0
            fast_u = np.mean(u[u > 0]) + (np.max(u) - np.mean(u[u > 0])) * fast_u_tresh
            width_ratio = len(u[u > fast_u]) / len(u[u > 0])

            outs = [float(_) for _ in outs]

            target = float(sum(outs[0:4])) / float(sum(outs[4:7]))
            if (curl > 30000) or ((width_ratio < 0.25) or (width_ratio > 0.43)):
                print('Speed common condition violated')
                target = 0

            mean_diff = float(np.mean([abs(float(o) / np.mean(outs[0:4]) - 1) * 100 for o in outs[0:4]]))
            if USE_AVG_CONST and any([abs(float(o) / np.mean(outs[0:4]) - 1) * 100 > 5.0 for o in outs[0:4]]):
                print('Speed equality violated', [abs(float(o) / np.mean(outs[0:4]) - 1) * 100 for o in outs[0:4]])
                target = 0

            if target > 0:
                print(round(target, 4), round(mean_diff, 2), [round(_, 4) for _ in outs], round(float(curl)),
                      round(curv, 4), round(width_ratio, 4))

            self.client.clear()

        else:
            print(f'Cached: {target}')

        return -target

    def _poly_add(self, model, polygons):
        for n, poly in enumerate(polygons):
            try:
                model.java.component('comp1').geom('geom1').create('pol' + str(n + 1), 'Polygon')
            except Exception:
                pass
            model.java.component('comp1').geom('geom1').feature('pol' + str(n + 1)).set('x', poly[0])
            model.java.component('comp1').geom('geom1').feature('pol' + str(n + 1)).set('y', poly[1])
        return model

    def _save_simulation_result(self, configuration, model):
        if not os.path.exists('./models'):
            os.mkdir('./models')
        model_uid = str(uuid4())
        model.save(f'./models/{model_uid}.mph')
        db = pickledb.load('comsol_db.saved', False)
        db.set(str(configuration), model_uid)
        db.dump()

        if not os.path.exists('./structures'):
            os.mkdir('./structures')

        with open(f'./structures/{model_uid}.str', 'wb') as f:
            pickle.dump(configuration, f)

        return model_uid

    def _load_simulation_result(self, client, configuration):
        db = pickledb.load('comsol_db.saved', False)

        model_uid = db.get(str(configuration))

        if model_uid is False:
            return None, None

        model = client.load(f'./models/{model_uid}.mph')

        return model, model_uid

    def _save_fitness(self, configuration, fitness):
        array_structure = [[[p.x, p.y] for p in poly.points] for poly in configuration.polygons]
        db = pickledb.load('fitness_db.saved', False)
        db.set(str(array_structure), str(round(fitness, 4)))
        db.dump()

    def _load_fitness(self, configuration):
        array_structure = [[[p.x, p.y] for p in poly.points] for poly in configuration.polygons]
        db = pickledb.load('fitness_db.saved', False)

        db_models = pickledb.load('comsol_db.saved', False)

        model_uid = db_models.get(str(configuration))

        result = db.get(str(array_structure))

        if result is False:
            return None, None
        else:
            fitness = float(result)

        return float(fitness), model_uid
