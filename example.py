import logging
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import random
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - Step %(step)d - %(message)s",
    handlers=[logging.StreamHandler()],
)


# Worker Agent
class Worker(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.income = 10  # Hourly wage
        self.savings = 0
        self.needs = {
            "food": 1,
            "clothing": 30,
            "heavy_industry": 90,
        }  # Demand frequency in steps
        self.last_purchase = {"food": 0, "clothing": 0, "heavy_industry": 0}

    def step(self):
        # Earn income and split between savings and consumption
        earnings = self.income
        self.savings += earnings * 0.5  # 50% to savings
        consumption_budget = earnings * 0.5  # 50% to spend
        self.model.bank.deposit(self, earnings * 0.5)

        # Consumption behavior based on need frequency
        for product, frequency in self.needs.items():
            if (self.model.schedule.steps - self.last_purchase[product]) >= frequency:
                price = self.model.market.prices[product]
                if consumption_budget >= price:
                    self.model.market.buy(self, product, 1)
                    consumption_budget -= price
                    self.last_purchase[product] = self.model.schedule.steps


# Factory Agent
class Factory(Agent):
    def __init__(self, unique_id, model, product_type):
        super().__init__(unique_id, model)
        self.product_type = product_type  # "food", "clothing", or "heavy_industry"
        self.workers = 5  # Initial number of workers
        self.inventory = 0
        self.production_rate = 2  # Units produced per worker per step
        self.max_workers = 10

    def step(self):
        # Produce goods
        production = self.workers * self.production_rate
        self.inventory += production

        # Adjust workforce based on market demand
        demand = self.model.market.demand[self.product_type]
        if demand > self.inventory and self.workers < self.max_workers:
            self.workers += 1  # Hire
        elif demand < self.inventory and self.workers > 1:
            self.workers -= 1  # Fire
        # Random workforce fluctuation
        if random.random() < 0.1:  # 10% chance to adjust workers randomly
            self.workers += (
                random.choice([-1, 1]) if 1 < self.workers < self.max_workers else 0
            )

        # Sell to market
        sold = min(self.inventory, demand)
        self.inventory -= sold
        self.model.market.supply[self.product_type] += sold


# Market Agent
class Market(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.prices = {"food": 5, "clothing": 20, "heavy_industry": 50}
        self.supply = {"food": 0, "clothing": 0, "heavy_industry": 0}
        self.demand = {"food": 0, "clothing": 0, "heavy_industry": 0}

    def buy(self, worker, product, quantity):
        self.demand[product] += quantity

    def step(self):
        # Update prices based on supply and demand
        for product in self.prices:
            if self.supply[product] > self.demand[product]:
                self.prices[product] *= 0.95  # Price drops
            elif self.supply[product] < self.demand[product]:
                self.prices[product] *= 1.05  # Price rises
            self.supply[product] = 0  # Reset supply after step
            self.demand[product] = 0  # Reset demand after step


# Bank Agent
class Bank(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.total_savings = 1000  # Initial wealth
        self.interest_rate = 0.01

    def deposit(self, worker, amount):
        self.total_savings += amount

    def step(self):
        # Pay interest on savings
        self.total_savings *= 1 + self.interest_rate
        if self.total_savings <= 0:
            logging.info(
                "Bank has gone bankrupt!", extra={"step": self.model.schedule.steps}
            )
            self.model.running = False


# Economic Model
class EconomicModel(Model):
    def __init__(self, width=10, height=10, num_workers=20):
        self.num_workers = num_workers
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(width, height, True)
        self.running = True

        # Create agents
        self.market = Market(0, self)
        self.schedule.add(self.market)
        self.bank = Bank(1, self)
        self.schedule.add(self.bank)

        # Factories
        for i, product in enumerate(["food", "clothing", "heavy_industry"], start=2):
            factory = Factory(i, self, product)
            self.schedule.add(factory)
            self.grid.place_agent(
                factory, (random.randint(0, width - 1), random.randint(0, height - 1))
            )

        # Workers
        for i in range(self.num_workers):
            worker = Worker(i + 5, self)
            self.schedule.add(worker)
            self.grid.place_agent(
                worker, (random.randint(0, width - 1), random.randint(0, height - 1))
            )

        # Data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Unemployment": lambda m: sum(
                    1
                    for a in m.schedule.agents
                    if isinstance(a, Worker) and a.income == 0
                ),
                "Food_Price": lambda m: m.market.prices["food"],
                "Clothing_Price": lambda m: m.market.prices["clothing"],
                "Heavy_Industry_Price": lambda m: m.market.prices["heavy_industry"],
                "Food_Inventory": lambda m: next(
                    a.inventory
                    for a in m.schedule.agents
                    if isinstance(a, Factory) and a.product_type == "food"
                ),
                "Total_Savings": lambda m: m.bank.total_savings,
                "Average_Worker_Savings": lambda m: np.mean(
                    [a.savings for a in m.schedule.agents if isinstance(a, Worker)]
                ),
            }
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        # Log key metrics every 10 steps
        if self.schedule.steps % 10 == 0:
            data = self.datacollector.get_model_vars_dataframe().iloc[-1]
            logging.info(
                f"Unemployment: {data['Unemployment']}, Food Price: {data['Food_Price']:.2f}, "
                f"Clothing Price: {data['Clothing_Price']:.2f}, Heavy Industry Price: {data['Heavy_Industry_Price']:.2f}, "
                f"Food Inventory: {data['Food_Inventory']:.2f}, Total Savings: {data['Total_Savings']:.2f}, "
                f"Avg Worker Savings: {data['Average_Worker_Savings']:.2f}",
                extra={"step": self.schedule.steps},
            )


# Run the simulation
if __name__ == "__main__":
    model = EconomicModel()
    max_steps = 100  # Run for 1000 steps
    for step in range(max_steps):
        model.step()
        if not model.running:
            break
    logging.info("Simulation ended.", extra={"step": model.schedule.steps})
