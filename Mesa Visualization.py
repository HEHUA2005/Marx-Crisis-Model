from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.datacollection import DataCollector
import random

# --------------------------
# Agent Definitions
# --------------------------
class SocialAgent(Agent):
    def __init__(self, unique_id, model, agent_type, wealth=None, generation=0):
        super().__init__(unique_id, model)
        self.agent_type = agent_type
        self.generation = generation
        if wealth is not None:
            self.wealth = wealth
        else:
            self.wealth = random.randint(4000, 6000) if agent_type == 'bourgeoisie' else random.randint(1, 5)
        self.reproduce_timer = 0
        self.age = 0
        self.max_age = random.randint(100, 150) if agent_type == 'bourgeoisie' else random.randint(60, 100)

    def step(self):
        self.age += 1
        # 如果代理死亡，移除它
        if self.age >= self.max_age:
            if self.pos:
                self.model.grid.remove_agent(self)
                self.model.schedule.remove(self)
                self.model.occupied_positions.discard(self.pos)
            return

        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)

        if self.agent_type == "bourgeoisie":
            proletariat_neighbors = [n for n in neighbors if n.agent_type == "proletariat"]
            for neighbor in proletariat_neighbors:
                if neighbor.wealth > 0:
                    neighbor.wealth -= 1
                    multiplier = 1.0 + 0.1 * self.generation
                    self.wealth += int(100 * multiplier)
                else:
                    self.wealth = max(0, self.wealth - 10)

            if self.wealth > 10000:
                self.wealth -= 100

            if self.wealth > 20000:
                redistribution_amount = self.wealth // 2
                self.wealth -= redistribution_amount
                proletariat_agents = [a for a in self.model.schedule.agents if a.agent_type == 'proletariat']
                for agent in proletariat_agents:
                    agent.wealth += redistribution_amount // len(proletariat_agents)

        elif self.agent_type == "proletariat":
            empty_cells = [cell for cell in self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
                           if self.model.grid.is_cell_empty(cell)]
            if empty_cells:
                new_pos = self.random.choice(empty_cells)
                self.model.grid.move_agent(self, new_pos)

            fellow_proletariats = [n for n in neighbors if n.agent_type == "proletariat"]
            if len(fellow_proletariats) >= 3:
                self.wealth += 1
                for neighbor in fellow_proletariats:
                    neighbor.wealth += 1

            if self.wealth >= 100:
                self.agent_type = "bourgeoisie"
                self.wealth = random.randint(4000, 6000)
                self.max_age = random.randint(100, 150)
                self.model.promotions += 1  # 记录晋升人数

        self.reproduce_timer += 1
        if self.reproduce_timer >= 10:
            self.reproduce_timer = 0
            if self.agent_type == "proletariat":
                num_children = random.choice([1, 2, 3])
            else:
                num_children = random.choice([0, 1])

            for _ in range(num_children):
                new_id = self.model.next_id()
                child_type = self.agent_type
                child_wealth = self.wealth // 10 if child_type == "bourgeoisie" else random.randint(1, 3)
                child = SocialAgent(new_id, self.model, child_type, wealth=child_wealth, generation=self.generation + 1)
                self.model.schedule.add(child)
                empty = [cell for cell in self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
                         if self.model.grid.is_cell_empty(cell)]
                if empty:
                    pos = self.random.choice(empty)
                    self.model.grid.place_agent(child, pos)
                    self.model.occupied_positions.add(pos)

# --------------------------
# Model Definition
# --------------------------
class ClassConflictModel(Model):
    def __init__(self, width, height, num_proletariat, num_bourgeoisie):
        self.grid = MultiGrid(width, height, torus=True)
        self.schedule = RandomActivation(self)
        self.occupied_positions = set()
        self.agent_count = 0

        # 记录晋升和降级
        self.promotions = 0
        self.demotions = 0

        self.datacollector = DataCollector(
            model_reporters={
                "ProletariatRatio": self.get_proletariat_ratio,
                "WealthRatio": self.get_wealth_ratio,
                "Proletariat": self.get_num_proletariat,
                "Bourgeoisie": self.get_num_bourgeoisie,
                "Proletariat Average Wealth": self.get_average_proletariat_wealth,
                "Bourgeoisie Average Wealth": self.get_average_bourgeoisie_wealth,
                "Promotions": lambda m: m.promotions,
                "Demotions": lambda m: m.demotions
            }
        )

        for _ in range(num_proletariat):
            agent = SocialAgent(self.next_id(), self, "proletariat")
            self.add_agent(agent)

        for _ in range(num_bourgeoisie):
            agent = SocialAgent(self.next_id(), self, "bourgeoisie")
            self.add_agent(agent)

    def add_agent(self, agent):
        self.schedule.add(agent)
        while True:
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            if (x, y) not in self.occupied_positions:
                self.occupied_positions.add((x, y))
                break
        self.grid.place_agent(agent, (x, y))

    def next_id(self):
        self.agent_count += 1
        return self.agent_count

    def step(self):
        # 在每次步骤中清理死亡代理
        dead_agents = [agent for agent in self.schedule.agents if agent.age >= agent.max_age]
        for agent in dead_agents:
            self.schedule.remove(agent)
            self.grid.remove_agent(agent)
            self.occupied_positions.discard(agent.pos)

        self.schedule.step()
        self.datacollector.collect(self)
        self.promotions = 0
        self.demotions = 0

    def get_num_proletariat(self):
        return sum(1 for agent in self.schedule.agents if agent.agent_type == "proletariat")

    def get_num_bourgeoisie(self):
        return sum(1 for agent in self.schedule.agents if agent.agent_type == "bourgeoisie")

    def get_average_proletariat_wealth(self):
        proletariats = [agent for agent in self.schedule.agents if agent.agent_type == "proletariat"]
        return sum(p.wealth for p in proletariats) / len(proletariats) if proletariats else 0

    def get_average_bourgeoisie_wealth(self):
        bourgeoisie = [agent for agent in self.schedule.agents if agent.agent_type == "bourgeoisie"]
        return sum(b.wealth for b in bourgeoisie) / len(bourgeoisie) if bourgeoisie else 0

    def get_proletariat_ratio(self):
        total = len(self.schedule.agents)
        return self.get_num_proletariat() / total if total > 0 else 0

    def get_wealth_ratio(self):
        total_wealth = sum(agent.wealth for agent in self.schedule.agents)
        proletariat_wealth = sum(agent.wealth for agent in self.schedule.agents if agent.agent_type == "proletariat")
        return proletariat_wealth / total_wealth if total_wealth > 0 else 0

# --------------------------
# Visualization
# --------------------------
def agent_portrayal(agent):
    if agent.agent_type == "proletariat":
        color = "red"
    else:
        color = "blue"

    size = 0.2 + 0.00005 * agent.wealth
    return {
        "Shape": "circle",
        "Filled": "true",
        "Color": color,
        "Layer": 0,
        "r": min(size, 1.0)
    }

grid = CanvasGrid(agent_portrayal, 20, 20, 800, 800)

proletariat_chart = ChartModule(
    [
        {"Label": "Proletariat", "Color": "red"},
        {"Label": "Bourgeoisie", "Color": "blue"}
    ]
)

wealth_chart = ChartModule(
    [
        {"Label": "Proletariat Average Wealth", "Color": "red"},
        {"Label": "Bourgeoisie Average Wealth", "Color": "blue"}
    ]
)

ratio_chart = ChartModule(
    [
        {"Label": "ProletariatRatio", "Color": "red"},
        {"Label": "WealthRatio", "Color": "blue"}
    ]
)

# 阶级流动图
class_mobility_chart = ChartModule(
    [
        {"Label": "Promotions", "Color": "green"},
        {"Label": "Demotions", "Color": "orange"}
    ]
)

server = ModularServer(
    ClassConflictModel,
    [grid, proletariat_chart, wealth_chart, ratio_chart, class_mobility_chart],
    "Class Conflict Model",
    {"width": 20, "height": 20, "num_proletariat": 100, "num_bourgeoisie": 10}
)

server.launch()



