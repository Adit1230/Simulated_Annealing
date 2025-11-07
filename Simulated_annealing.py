import math
import random
from typing import Callable

class State():
    def __init__(self):
        pass

    def cost(self, state) -> float:
        pass

    def get_neighbour(self) -> tuple[int, float]:
        pass

    def update(self, idx : int, change : float) -> None:
        pass

    def cost_change(self, idx : int, change : float) -> float:
        pass


class Annealer():
    def __init__(self, state : State, initial_temp : float, temperature_schedule : str | Callable[[float, int], float], scheduling_constant : float):
        self.state = state
        self.temperature = initial_temp
        self.step = 0
        self.temperature_schedule = temperature_schedule
        self.scheduling_constant = scheduling_constant

    def calc_prob(self, energy_change : float) -> float:
        prob = math.exp(- energy_change / self.temperature)
        return prob
    
    def linear_schedule(self) -> None:
        if (self.temperature >= self.scheduling_constant):
            self.temperature -= self.scheduling_constant
        else:
            self.temperature = 0
    
    def exponential_schedule(self) -> None:
        self.temperature = (1 - self.scheduling_constant) * self.temperature
    
    def logarithmic_schedule(self):
        self.temperature = self.scheduling_constant / math.log(self.step + 1)
    
    def schedule_step(self) -> None:
        if self.temperature_schedule == 'linear':
            self.linear_schedule()
        elif self.temperature_schedule == 'exponential':
            self.exponential_schedule()
        elif self.temperature_schedule == 'logarithmic':
            self.logarithmic_schedule()
        else:
            self.temperature = self.temperature_schedule(self.temperature, self.step)
    
    def anneal_step(self) -> bool:
        neighbour_idx, neighbour_change = self.state.get_neighbour()
        del_E = self.state.cost_change(neighbour_idx, neighbour_change)

        changed = False
        
        if (del_E <= 0):
            self.state.update(neighbour_idx, neighbour_change)
            changed = True
        else:
            prob = self.calc_prob(del_E)
            if (random.random() <= prob):
                self.state.update(neighbour_idx, neighbour_change)
                changed = True
        
        self.step += 1
        self.schedule_step()

        return changed
    
    def anneal(self, steps = None, stop_temp = None, unchanged_threshold = 100) -> State:
        initial_step = self.step
        unchanged = 0
        changed = False

        while True:
            changed = self.anneal_step()

            if changed:
                unchanged = 0
            else:
                unchanged += 1

            if (steps is not None) and ((self.step - initial_step) >= steps):
                break
            elif (stop_temp is not None) and (self.temperature <= stop_temp):
                break
            elif (unchanged >= unchanged_threshold):
                break
        
        return self.state
    
    def get_solution(self) -> State:
        return self.state
