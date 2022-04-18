from gefest.core.opt.GA.base_GA import BaseGA
from gefest.core.opt.analytics import EvoAnalytics


class GA(BaseGA):

    def solution(self, verbose=True, **kwargs):
        self.generation_number = 0

        self.fitness()
        best = None

        while self.generation_number <= self.params.max_gens:
            print(f'Generation {self.generation_number}')

            un_pop = set()
            self._pop = \
                [un_pop.add(str(ind.genotype)) or ind for ind in self._pop
                 if str(ind.genotype) not in un_pop]

            self._pop.extend(self.reproduce(self._pop))

            for individ in self._pop:
                individ.population_number = self.generation_number

            self.fitness()

            selected = self.tournament_selection()
            self._pop = sorted(selected, key=lambda x: x.fitness)
            best = sorted(self._pop, key=lambda x: x.fitness)[0]

            print(f'Best fitness is {best.fitness}')

            self.generation_number += 1

        return self._pop, best
