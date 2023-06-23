import itertools
import time
from copy import deepcopy
import pickle
import matplotlib.pyplot as plt


from gefest.core.structure.structure import Structure
from gefest.core.structure.point import Point
from gefest.core.algs.geom.validation import out_of_bound, too_close, intersection
from cases.sensitivity_analysis.configuration_sa import sa_surrogate_estimator
from cases.breakwaters.configuration_de import bw_domain

from cases.sensitivity_analysis.creator_structures import get_structure
from cases.main_conf import opt_params


domain, task_setup = bw_domain.configurate_domain(
    poly_num=opt_params.n_polys,
    points_num=opt_params.n_points,
    is_closed=opt_params.is_closed,
)

estimator = sa_surrogate_estimator.configurate_estimator(domain=domain)


class SA_methods:
    def __init__(self):
        self.optimized_structure = optimized_pop
        self.cost = estimator.estimate
        self.input_domain = domain
        self.sa_time_history = [0]
        self.start_time = time.time()

    @property
    def get_time_history(self):
        return self.sa_time_history

    def moving_position(self):
        structure = self.optimized_structure
        for poly_num, poly in enumerate(structure.polygons):
            poly.id = "poly_" + str(poly_num)
        init_fitnes = round(
            self.cost([structure])[0], 3
        )  # only high of wave in multicreterial loss

        fitnes_history = []
        structure_history = []
        polygon_history = []
        fitnes_history.append(init_fitnes)
        structure_history.append(structure)
        polygon_history.append(f"init_moving, fitnes={init_fitnes}")
        current_fitnes = init_fitnes

        for poly_num, poly in enumerate(structure.polygons):
            step_fitnes = 0
            max_attempts = 3

            if poly.id != "fixed":
                moving_step = self.input_domain.geometry.get_length(polygon=poly) * 0.2

                while step_fitnes <= current_fitnes and max_attempts > 0:
                    step_structure, step_fitnes, worse_res = self._moving_for_one_step(
                        structure=structure,
                        poly_number=poly_num,
                        moving_step=moving_step,
                        init_fitnes=current_fitnes,
                    )
                    structure_history.append(step_structure)
                    fitnes_history.append(step_fitnes)
                    end_step_time = time.time()
                    self.sa_time_history.append(end_step_time - self.start_time)

                    if worse_res:
                        fitnes_diff = round(
                            100 * ((worse_res - current_fitnes) / current_fitnes), 1
                        )
                        polygon_history.append(
                            f"{str(poly.id)}, step={round(moving_step)},\
                                                fitnes=+{str(fitnes_diff)}%"
                        )
                    else:
                        fitnes_diff = round(
                            100 * ((step_fitnes - current_fitnes) / current_fitnes), 1
                        )
                        polygon_history.append(
                            f"{str(poly.id)}, step={round(moving_step)},\
                                                fitnes={str(fitnes_diff)}%"
                        )

                    if step_fitnes >= current_fitnes:
                        max_attempts -= 1
                        moving_step = moving_step / 2
                    else:
                        current_fitnes = step_fitnes
                        structure = deepcopy(step_structure)

        return fitnes_history, structure_history, polygon_history

    def _moving_for_one_step(
        self, structure: Structure, poly_number: int, moving_step, init_fitnes
    ):
        moved_init_poly = structure.polygons[poly_number]
        directions = ["north", "south", "east", "west", "n-w", "s-w", "n-e", "s-e"]
        results = {}
        worse_results = {}

        for direct in directions:
            moved_poly = deepcopy(moved_init_poly)
            for idx, point in enumerate(moved_poly.points):
                moved_poly.points[idx] = self._moving_point(direct, point, moving_step)

            tmp_structure = deepcopy(structure)
            tmp_structure.polygons[poly_number] = moved_poly
            fitnes = round(self.cost([tmp_structure])[0], 3)
            non_unvalid = not any(
                [
                    out_of_bound(tmp_structure, self.input_domain),
                    too_close(tmp_structure, self.input_domain),
                    intersection(tmp_structure, self.input_domain),
                ]
            )
            if fitnes < init_fitnes and non_unvalid:
                results[fitnes] = tmp_structure
            elif fitnes >= init_fitnes and non_unvalid:
                worse_results[fitnes] = tmp_structure
            else:
                worse_results[init_fitnes] = tmp_structure

        if results:
            best_structure = results[min(results)]
            best_fitnes = min(results)
            return best_structure, best_fitnes, 0
        else:
            best_worse_fitnes = min(worse_results)
            return structure, init_fitnes, best_worse_fitnes

    def _moving_point(self, direction: str, point: Point, moving_step) -> Point:
        directions = {
            "north": Point(point.x + moving_step, point.y),
            "south": Point(point.x - moving_step, point.y),
            "east": Point(point.x, point.y + moving_step),
            "west": Point(point.x, point.y - moving_step),
            "n-w": Point(point.x + moving_step, point.y - moving_step),
            "s-w": Point(point.x - moving_step, point.y + moving_step),
            "n-e": Point(point.x + moving_step, point.y + moving_step),
            "s-e": Point(point.x - moving_step, point.y - moving_step),
        }
        return directions[direction]

    def exploring_combinations(self, structure: Structure, init_fitnes):
        current_fitnes = init_fitnes

        length_strcuture = []
        real_fitnes = []

        fitnes_history = []
        structure_history = []
        polygon_history = []

        fitnes_history.append(init_fitnes)
        structure_history.append(structure)
        polygon_history.append(f"init_combinations, fitnes={init_fitnes}")
        end_step_time = time.time()
        self.sa_time_history.append(end_step_time - self.start_time)

        for length in range(1, len(structure.polygons) + 1):
            for unique_comb in itertools.combinations(structure.polygons, length):
                tmp_structure = deepcopy(structure)
                tmp_structure.polygons = list(unique_comb)
                fitnes = round(self.cost([tmp_structure])[0], 3)
                real_fitnes.append(fitnes)

                length_polygons = [
                    self.input_domain.geometry.get_length(polygon=poly)
                    for poly in tmp_structure.polygons
                ]
                length_strcuture.append(sum(length_polygons))

                if fitnes < current_fitnes:
                    fitnes_diff = round(
                        100 * ((fitnes - current_fitnes) / current_fitnes), 1
                    )
                    current_fitnes = fitnes
                    structure_history.append(tmp_structure)
                    fitnes_history.append(fitnes)
                    ids = []
                    for polygon in tmp_structure.polygons:
                        ids.append(polygon.id)
                    polygon_history.append(f"{str(ids)}, fitnes={str(fitnes_diff)}%")
                else:
                    structure_history.append(tmp_structure)
                    fitnes_history.append(current_fitnes)
                    fitnes_diff = round(
                        100 * ((fitnes - current_fitnes) / current_fitnes), 1
                    )
                    ids = []
                    for polygon in tmp_structure.polygons:
                        ids.append(polygon.id)
                    polygon_history.append(f"{str(ids)}, fitnes=+{str(fitnes_diff)}%")

                end_step_time = time.time()
                self.sa_time_history.append(end_step_time - self.start_time)

        best_indexes = [i for i, x in enumerate(real_fitnes) if x <= current_fitnes]
        struct_lengths = {idx: length_strcuture[idx] for idx in best_indexes}
        best_index = min(struct_lengths, key=struct_lengths.get)

        structure_history.append(structure_history[best_index])
        fitnes_history.append(fitnes_history[best_index])
        polygon_history.append(polygon_history[best_index])

        return fitnes_history, structure_history, polygon_history

    def removing_points(self, structure: Structure, init_fitnes):
        fitnes_history = []
        structure_history = []
        polygon_history = []

        fitnes_history.append(init_fitnes)
        structure_history.append(structure)
        polygon_history.append(f"init_removing_points, fitnes={init_fitnes}")
        end_step_time = time.time()
        self.sa_time_history.append(end_step_time - self.start_time)

        current_fitnes = init_fitnes

        for poly_number, polygon in enumerate(structure.polygons):
            if len(polygon.points) > 2:
                closed = bool(polygon.points[0] == polygon.points[-1])

                if closed:
                    for idx_point in range(1, len(polygon.points)):
                        exploring_polygon = deepcopy(polygon)
                        tmp_structure = deepcopy(structure)

                        exploring_point_coords = (
                            tmp_structure.polygons[poly_number]
                            .points[idx_point]
                            .coords()
                        )
                        exploring_polygon.points.pop(idx_point)
                        tmp_structure.polygons[poly_number] = exploring_polygon

                        fitnes = round(self.cost([tmp_structure])[0], 3)
                        fitnes_diff = round(
                            100 * ((fitnes - current_fitnes) / current_fitnes), 1
                        )

                        if fitnes < current_fitnes:
                            current_fitnes = fitnes
                            structure_history.append(tmp_structure)
                            fitnes_history.append(fitnes)
                            polygon_history.append(
                                f"{str(polygon.id)}, del={str(exploring_point_coords)},\
                                                    fitnes={str(fitnes_diff)}%"
                            )
                        else:
                            structure_history.append(tmp_structure)
                            fitnes_history.append(current_fitnes)
                            polygon_history.append(
                                f"{str(polygon.id)}, del={str(exploring_point_coords)},\
                                                    fitnes=+{str(fitnes_diff)}%"
                            )
                        end_step_time = time.time()
                        self.sa_time_history.append(end_step_time - self.start_time)

                else:
                    for idx, point in enumerate(polygon.points):
                        exploring_polygon = deepcopy(polygon)
                        tmp_structure = deepcopy(structure)

                        exploring_point_coords = point.coords()
                        exploring_polygon.points.pop(idx)
                        tmp_structure.polygons[poly_number] = exploring_polygon

                        fitnes = round(self.cost([tmp_structure])[0], 3)
                        fitnes_diff = round(
                            100 * ((fitnes - current_fitnes) / current_fitnes), 1
                        )

                        if fitnes < current_fitnes:
                            current_fitnes = fitnes
                            structure_history.append(tmp_structure)
                            fitnes_history.append(fitnes)
                            polygon_history.append(
                                f"{str(polygon.id)}, del={str(exploring_point_coords)},\
                                                    fitnes={str(fitnes_diff)}%"
                            )
                        else:
                            structure_history.append(tmp_structure)
                            fitnes_history.append(current_fitnes)
                            polygon_history.append(
                                f"{str(polygon.id)}, del={str(exploring_point_coords)},\
                                                    fitnes=+{str(fitnes_diff)}%"
                            )
                        end_step_time = time.time()
                        self.sa_time_history.append(end_step_time - self.start_time)

        best_fitnes = min(fitnes_history)
        best_idx = fitnes_history.index(best_fitnes)
        best_structure = structure_history[best_idx]

        fitnes_history.append(best_fitnes)
        structure_history.append(best_structure)
        polygon_history.append(f"best_structure after removing points")
        end_step_time = time.time()
        self.sa_time_history.append(end_step_time - self.start_time)

        return fitnes_history, structure_history, polygon_history

    def rotate_objects(self, structure: Structure, init_fitnes: int):
        rotate_func = self.input_domain.geometry.rotate_poly
        fitnes_history = []
        structure_history = []
        polygon_history = []

        fitnes_history.append(init_fitnes)
        structure_history.append(structure)
        polygon_history.append(f"init_rotates, fitnes={init_fitnes}")
        end_step_time = time.time()
        self.sa_time_history.append(end_step_time - self.start_time)

        curent_fitnes = init_fitnes

        for poly_num, poly in enumerate(structure.polygons):
            tmp_fit_history = []
            tmp_str_history = []

            angles = [angle for angle in range(45, 360, 45)]

            if poly.id != "fixed":
                for angle in angles:
                    tmp_structure = deepcopy(structure)
                    rotated_poly = deepcopy(poly)

                    rotated_poly = rotate_func(rotated_poly, angle=angle)
                    tmp_structure.polygons[poly_num] = rotated_poly

                    fitnes = round(self.cost([tmp_structure])[0], 3)
                    tmp_fit_history.append([fitnes, angle])
                    tmp_str_history.append(tmp_structure)

                best_poly_fit = min(tmp_fit_history)
                idx_best = tmp_fit_history.index(best_poly_fit)
                fitnes_diff = round(
                    100 * ((best_poly_fit[0] - curent_fitnes) / curent_fitnes), 1
                )

                if best_poly_fit[0] < curent_fitnes:
                    curent_fitnes = best_poly_fit[0]
                    best_tmp_structure = tmp_str_history[idx_best]
                    structure.polygons[poly_num] = best_tmp_structure.polygons[poly_num]
                    fitnes_history.append(best_poly_fit[0])
                    structure_history.append(best_tmp_structure)
                    polygon_history.append(
                        f"{str(poly.id)}, best_angle={best_poly_fit[1]}, fitness={fitnes_diff}%"
                    )
                else:
                    fitnes_history.append(curent_fitnes)
                    structure_history.append(tmp_str_history[idx_best])
                    polygon_history.append(
                        f"{str(poly.id)}, best_angle={best_poly_fit[1]}, fitness=+{fitnes_diff}%"
                    )
                end_step_time = time.time()
                self.sa_time_history.append(end_step_time - self.start_time)

        best_fitnes = min(fitnes_history)
        best_idx = fitnes_history.index(best_fitnes)
        best_structure = structure_history[best_idx]

        fitnes_history.append(best_fitnes)
        structure_history.append(best_structure)
        polygon_history.append(f"best_structure after rotating polygons")

        end_step_time = time.time()
        self.sa_time_history.append(end_step_time - self.start_time)

        return fitnes_history, structure_history, polygon_history


class SA(SA_methods):
    def __init__(self):
        super().__init__()

    def analysis(self):
        mov_fitnes, mov_structure, mov_poly = self.moving_position()
        rotated_fitnes, rotated_structure, rotated_poly = self.rotate_objects(
            mov_structure[-1], mov_fitnes[-1]
        )
        del_fitnes, del_structure, del_poly = self.exploring_combinations(
            rotated_structure[-1], rotated_fitnes[-1]
        )
        rm_points_fitnes, rm_points_structure, rm_points_poly = self.removing_points(
            del_structure[-1], del_fitnes[-1]
        )

        fitnes_history = mov_fitnes + rotated_fitnes + del_fitnes + rm_points_fitnes
        structure_history = (
            mov_structure + rotated_structure + del_structure + rm_points_structure
        )
        poly_history = mov_poly + rotated_poly + del_poly + rm_points_poly

        return fitnes_history, structure_history, poly_history


if __name__ == "__main__":
    number_experiments = 30
    evo_steps = 80
    pop_size = 15
    for i in range(number_experiments):
        full_archive = {}

        step_for_start_sa = [0.25, 0.5, 0.75, 1]
        step_for_start_sa = [
            round(evo_steps * percent_step) - 1 for percent_step in step_for_start_sa
        ]
        # root = project_root()
        best_evo_structure = get_structure(n_steps=evo_steps, pop_size=pop_size)

        with open(f"HistoryFiles/time_history.pickle", "rb") as f:
            evo_time_history = pickle.load(f)
            f.close()
        full_archive["evo_time_history"] = evo_time_history

        evo_fitnes_history = []
        for step in range(evo_steps):
            with open(f"HistoryFiles/performance_{step}.pickle", "rb") as f:
                step_fit = pickle.load(f)
                evo_fitnes_history.append(step_fit[0])
                f.close()
        full_archive["evo_fitnes_history"] = evo_fitnes_history

        for step_start in step_for_start_sa:
            step_archive = {}
            with open(f"HistoryFiles/population_{step_start}.pickle", "rb") as f:
                population_for_start_sa = pickle.load(f)
                f.close()
            optimized_pop = population_for_start_sa[0]
            step_archive[f"initial_structure_{step_start}"] = optimized_pop

            sensitivity = SA()
            fitnes, structure, poly = sensitivity.analysis()

            sa_time_history = sensitivity.get_time_history
            sa_time_history = [
                time_sa + evo_time_history[step_start - 1]
                for time_sa in sa_time_history
            ]
            step_archive[f"sa_fitnes_{step_start}"] = fitnes
            step_archive[f"sa_structures_{step_start}"] = structure
            step_archive[f"sa_step_description_{step_start}"] = poly
            step_archive[f"sa_time_{step_start}"] = sa_time_history

            full_archive[f"sa_step_{step_start}"] = step_archive

        with open(f"HistorySA/sa_archive_{i}.pickle", "wb") as handle:
            pickle.dump(full_archive, handle, protocol=pickle.HIGHEST_PROTOCOL)

        plt.plot(evo_time_history, evo_fitnes_history, ".b-", label="GEFEST alg")
        plt.plot(sa_time_history, fitnes, ".r-", label="GEFEST+SA alg")
        plt.title(f"Comparing algorithms, SA started from {step_start} generation")
        plt.xlabel(f"spent time, sec (number of generations-{evo_steps}")
        plt.ylabel("value of loss function")
        plt.legend()
        plt.savefig(f"comparing_methods_startfrom{step_start}.png")

        fitnes_difference = round(100 * (fitnes[0] - fitnes[-1]) / fitnes[0], 1)
        descriptions = poly
        x = list(range(len(poly)))
        y = fitnes

        fig, axd = plt.subplot_mosaic(
            [["upper", "upper"], ["lower left", "lower right"]],
            figsize=(18, 18),
            height_ratios=[1, 2],
        )

        fig.suptitle(
            f"Sensitivity analysis started from {step_start} gen, spent={sa_time_history[-1]}sec,\
                      fitnes decreased by {fitnes_difference}%"
        )

        structure[0].plot(color="r", ax=axd["lower left"], legend=True)
        axd["lower left"].set_title(f"Initial structure, fitnes={round(fitnes[0], 3)}")
        structure[-1].plot(ax=axd["lower right"], legend=True)
        axd["lower right"].set_title(
            f"Processed structure, fitnes={round(fitnes[-1], 3)}"
        )

        axd["upper"].plot(fitnes, c="c")
        axd["upper"].scatter(x, y, marker="o", c="c")
        for idx, text in enumerate(descriptions):
            axd["upper"].annotate(text, (x[idx] + 0.01, y[idx] + 0.01), rotation=45.0)
        axd["upper"].set_xlabel("iteration of senitivity analysis")
        axd["upper"].set_ylabel("loss - height of waves")

        fig.tight_layout()
        fig.savefig(f"sa_fullway_from{step_start}.png")
