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
        self.optimal_state = self.state.state

    def calc_prob(self, energy_change : float) -> float:
        if self.temperature == 0:
            prob = 0
        else:
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
        
        if (del_E <= 0):
            self.state.update(neighbour_idx, neighbour_change)
        else:
            prob = self.calc_prob(del_E)
            if (random.random() <= prob):
                self.state.update(neighbour_idx, neighbour_change)
            else:
                del_E = 0
        
        self.step += 1
        self.schedule_step()

        return del_E

    
    def anneal(self, steps = None, stop_temp = None, unchanged_threshold = 100) -> State:
        initial_step = self.step
        cost = self.state.cost(None)
        best_cost = cost
        best_state = self.state.state

        unchanged_steps = 0

        while True:
            del_E = self.anneal_step()

            cost += del_E

            if cost < best_cost:
                best_cost = cost
                best_state = self.state.state
                unchanged_steps = 0
            else:
                unchanged_steps += 1

            if (steps is not None) and ((self.step - initial_step) >= steps):
                break
            elif (stop_temp is not None) and (self.temperature <= stop_temp):
                break
            elif (unchanged_steps >= unchanged_threshold):
                break
        
        self.optimal_state = best_state
        self.state.state = best_state
        
        return self.state
    
    def get_solution(self) -> State:
        return self.state
