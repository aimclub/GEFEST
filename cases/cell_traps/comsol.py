# # -*- coding: utf-8 -*-
# """
# Created on Wed Feb  3 12:49:07 2021
import gc
import os
import pickle
from typing import Tuple
from uuid import uuid4

import numpy as np
import pickledb

from comsol.polygen import poly_draw
from core.structure.structure import Structure

# @author: user
# """

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
from core.utils import project_root
from core.utils import GlobalEnv

global_env = GlobalEnv

USE_AVG_CONST = False


def poly_add(model, polygons):
    for n, poly in enumerate(polygons):
        try:
            model.java.component('comp1').geom('geom1').create('pol' + str(n + 1), 'Polygon')
        except Exception:
            pass
        model.java.component('comp1').geom('geom1').feature('pol' + str(n + 1)).set('x', poly[0])
        model.java.component('comp1').geom('geom1').feature('pol' + str(n + 1)).set('y', poly[1])
    return model


def execute(structure: Structure, with_vizualization=True) -> Tuple[float, float, str]:
    gc.collect()
    client = GlobalEnv().comsol_client
    target, mean_diff, idx = _load_fitness(structure)
    if target is None or GlobalEnv().full_save_load:
        model, idx = _load_simulation_result(structure)
        if model is None:
            poly_box = []
            print('Start COMSOL')
            for i, pol in enumerate(structure.polygons):
                poly_repr = []
                poly_repr.append(' '.join([str(pt.x) for pt in pol.points]))
                poly_repr.append(' '.join([str(pt.y) for pt in pol.points]))
                poly_box.append(poly_repr)

            model = client.load(f'./rbc-trap-setup.mph')

            try:
                model = poly_add(model, poly_box)

                model.build()
                model.mesh()
                model.solve()
            except Exception as ex:
                print(ex)
                client.clear()
                return 0.0, 100.0, idx

            idx = _save_simulation_result(structure, model)

        try:
            outs = [model.evaluate('vlct_1'),
                    model.evaluate('vlct_2'),
                    model.evaluate('vlct_3'),
                    model.evaluate('vlct_4'),
                    # model.evaluate('vlct_5'),
                    model.evaluate('vlct_side'),
                    model.evaluate('vlct_main')]
        except Exception as ex:
            print(ex)
            client.clear()
            return 0.0, 100.0, idx

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

        if with_vizualization and target > 0:
            poly_draw(model)

            if not os.path.exists('./tmp'):
                os.mkdir('./tmp')

            plt.savefig(f'./tmp/{target}.png')
            plt.clf()

        client.clear()

        _save_fitness(structure, target, mean_diff)
        if target > 0:
            print(round(target, 4), round(mean_diff, 2), [round(_, 4) for _ in outs], round(float(curl)),
                  round(curv, 4), round(width_ratio, 4))
    else:
        print(f'Cached: {target}')

    return target, mean_diff, idx


def _save_simulation_result(configuration, model):
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


def _load_simulation_result(configuration):
    db = pickledb.load('comsol_db.saved', False)

    model_uid = db.get(str(configuration))

    if model_uid is False:
        return None, None

    model = GlobalEnv().comsol_client.load(f'./models/{model_uid}.mph')

    return model, model_uid


def _save_fitness(configuration, fitness1, fitness2):
    db = pickledb.load('fitness_db.saved', False)
    db.set(str(configuration), f'{str(fitness1)}|{str(fitness2)}')
    db.dump()


def _load_fitness(configuration):
    db = pickledb.load('fitness_db.saved', False)

    db_models = pickledb.load('comsol_db.saved', False)

    model_uid = db_models.get(str(configuration))

    result = db.get(str(configuration))

    if result is False:
        return None, None, None
    else:
        fitness, mean_diff = result.split('|')

    return float(fitness), float(mean_diff), model_uid