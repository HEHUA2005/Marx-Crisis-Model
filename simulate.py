import mesa
import numpy as np
import logging
from enum import Enum
from datetime import datetime
import pandas as pd

# 设置日志系统
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("economic_simulation.log"), logging.StreamHandler()],
)


def get_distance(pos1, pos2):
    """计算两个位置之间的曼哈顿距离"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


class ProductType(Enum):
    FOOD = 0
    LIGHT_INDUSTRY = 1
    HEAVY_INDUSTRY = 2


class Worker(mesa.Agent):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.living = True
        self.home_pos = pos
        self.wealth = 1000  # 初始有1000块积蓄
        self.employed = False  # 初始没有工作
        self.factory = None
        self.work_duration = 0  # 工作时长
        self.health = 100  # 健康值
        self.happiness = 50  # 幸福值
        self.hunger = 0  # 饥饿值
        self.consumption_cycle = {
            ProductType.FOOD: 1,
            ProductType.LIGHT_INDUSTRY: 2,
            ProductType.HEAVY_INDUSTRY: np.random.randint(30, 90),
        }

        self.last_consumption = {pt: 0 for pt in ProductType}

    def __str__(self):
        return f"Worker {self.unique_id}: Wealth={self.wealth:.1f}, Employed={self.employed}"

    def choose_factory(self):
        if self.model.schedule.steps % 30 != 0 or not self.living:
            return
        factories = []
        for factory in self.model.factories:
            if not factory.bankrupt and factory.hiring:
                factories.append(factory)
        if not factories:
            if self.employed and self.factory.bankrupt:
                self.employed = False
                logging.info(
                    f"Worker {self.unique_id} lost job due to factory bankruptcy"
                )
            return
        print(f"Worker {self.unique_id} is looking for a factory")
        print(f"Current factory: {self.factory}")
        print(f"Current position: {self.home_pos}")
        distances = [get_distance(self.home_pos, f.pos) for f in factories]
        max_dist = max(distances) if distances else 1
        scores = [
            (f.wage * (1 - d / max_dist)) * (1 - f.inventory_ratio)
            for f, d in zip(factories, distances)
        ]

        new_factory = factories[np.argmax(scores)]
        if new_factory != self.factory:
            old_factory = self.factory
            self.factory = new_factory
            self.employed = True
            if old_factory:
                old_factory.workers.remove(self)
            new_factory.workers.append(self)
            logging.debug(
                f"Worker {self.unique_id} switched to factory {new_factory.unique_id}"
            )

    def work(self):
        self.hunger += 1
        if not self.employed or not self.living:
            return

        # 工作时间决策
        if self.wealth < 500:
            base_hours = np.clip(16 - (self.wealth / 100), 6, 16)
        else:
            base_hours = 6

        if self.health < 50:
            base_hours = min(8, base_hours)

        self.work_duration = np.random.normal(base_hours, 1)
        self.work_duration = np.clip(self.work_duration, 0, 16)

        # 获得收入
        income = self.work_duration * self.factory.wage
        self.wealth += income

        # 健康变化
        if self.work_duration > 12:
            self.health -= 3
        elif self.work_duration < 8:
            self.health += 1
        self.health = np.clip(self.health, 0, 100)
        if self.health < 20 and self.wealth > 5:
            self.wealth -= 5  # 花钱治病
        if self.health <= 0:
            self.living = False
            self.employed = False
            self.factory.workers.remove(self)
            logging.info(f"Worker {self.unique_id} died due to health issues")
            return

    def consume(self):
        # 消费时，优先购买食品
        # 没钱时减少轻工业购买和重工业购买,极端贫穷时不再购买重工业产品
        # 购买顺序：食品 -> 轻工业 -> 重工业
        # 消费会增加Happiness
        # 买吃的减少Hunger
        if not self.living:
            return
        base_wealth = self.model.market.prices[ProductType.FOOD] * 7
        ## 首先单独考虑食物
        if self.hunger > 0:
            price = self.model.market.prices[ProductType.FOOD]
            if self.wealth - price >= 0:
                self.wealth -= price
                self.model.market.sell(ProductType.FOOD, 1)
                self.hunger -= 1
                self.happiness += 5
                self.last_consumption[ProductType.FOOD] = self.model.schedule.steps
                logging.debug(
                    f"Worker {self.unique_id} bought FOOD at price {price:.1f}"
                )
            else:
                return

        for product in ProductType:
            if product == ProductType.FOOD:
                continue
            cycle = self.consumption_cycle[product]
            price = self.model.market.prices[product]
            if (
                self.model.schedule.steps - self.last_consumption[product]
            ) < cycle or self.wealth - price < base_wealth:
                continue
            if self.wealth - price >= base_wealth:
                self.wealth -= price
                self.model.market.sell(product, 1)
                self.last_consumption[product] = self.model.schedule.steps
                logging.debug(
                    f"Worker {self.unique_id} bought {product.name} at price {price:.1f}"
                )
                self.happiness += 25 * product

    def step(self):
        self.choose_factory()
        self.work()
        self.consume()


class Factory(mesa.Agent):
    def __init__(self, unique_id, model, pos, product_type):
        super().__init__(unique_id, model)
        self.product_type = product_type
        self.pos = pos
        self.wage = 15  # 不同行业工资初始化一样，根据市场需求来调整
        self.production = 100
        self.inventory = 100
        self.workers = []
        self.hiring = True
        self.debt = 0
        self.bankrupt = False
        self.last_month_sales = 100
        self.last_adjustment = 0
        self.job_offer = 3

    def __str__(self):
        status = (
            "BANKRUPT"
            if self.bankrupt
            else f"Production={self.production}, Workers={len(self.workers)}"
        )
        return f"Factory {self.unique_id} ({self.product_type.name}): {status}"

    @property
    def inventory_ratio(self):
        return self.inventory / (self.production * 30 + 1e-5)

    def adjust_production(self):
        if self.model.schedule.steps % 30 != 0:
            return

        sold_ratio = min(1, self.last_month_sales / (self.production + 1e-5))
        adjustment = 0.5 + 0.5 * sold_ratio

        old_production = self.production
        self.production = int(self.production * adjustment)

        # 工资调整
        if sold_ratio > 0.8:
            self.wage *= 1.05
        elif sold_ratio < 0.5:
            self.wage *= 0.95
        self.wage = max(5, min(50, self.wage))

        # 债务处理
        if self.debt > self.production * 10:
            self.debt *= 1.05  # 债务增长
            if np.random.random() < 0.1:  # 10%概率获得银行贷款
                self.debt *= 0.9

        logging.info(
            f"Factory {self.unique_id} adjusted production: {old_production} -> {self.production} "
            f"(Sold ratio: {sold_ratio:.2f}, Wage: {self.wage:.1f})"
        )

    def check_bankruptcy(self):
        if self.bankrupt:
            return

        if (
            self.inventory_ratio > 2.0 and self.debt > self.production * 15
        ) or self.debt > self.production * 30:
            self.bankrupt = True
            for worker in self.workers[:]:
                worker.employed = False
                self.workers.remove(worker)
            logging.warning(
                f"Factory {self.unique_id} went bankrupt! Debt: {self.debt:.1f}, Inventory: {self.inventory_ratio:.1f}"
            )

    def produce(self):
        if self.bankrupt:
            return

        self.inventory += self.production
        self.last_month_sales = max(0, self.last_month_sales)

    def step(self):
        self.adjust_production()
        self.check_bankruptcy()
        self.produce()


# class Bank(mesa.Agent):
#     def __init__(self, unique_id, model):
#         super().__init__(unique_id, model)
#         self.reserves = 100000
#         self.deposit_interest = 0.02
#         self.loan_interest = 0.05
#         self.bankrupt = False

#     def __str__(self):
#         status = "BANKRUPT" if self.bankrupt else f"Reserves={self.reserves:.1f}"
#         return f"Bank: {status}"

#     def check_solvency(self):
#         if self.reserves < 0:
#             self.bankrupt = True
#             logging.critical("!!! BANKING COLLAPSE !!! All deposits lost!")
#             for worker in self.model.workers:
#                 worker.bank_savings = 0

#     def step(self):
#         if self.bankrupt:
#             return

#         # 支付存款利息
#         total_deposits = sum(w.bank_savings for w in self.model.workers)
#         interest_payment = total_deposits * self.deposit_interest
#         self.reserves -= interest_payment

#         # 收回贷款
#         total_loans = sum(f.debt for f in self.model.factories)
#         if total_loans > 0:
#             repayment = min(total_loans * 0.1, self.reserves * 0.5)
#             for factory in self.model.factories:
#                 if factory.debt > 0:
#                     factory.debt -= repayment * (factory.debt / total_loans)
#             self.reserves += repayment

#         self.check_solvency()
#         logging.info(
#             f"Bank status: Reserves={self.reserves:.1f}, Loans={total_loans:.1f}"
#         )


class Government(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.last_intervention = -100

    def intervene(self):
        current_step = self.model.schedule.steps
        if current_step - self.last_intervention < 60:  # 至少60天间隔
            return
        print(f"model workers number: {len(self.model.workers)}")
        unemployment = sum(1 for w in self.model.workers if not w.employed) / len(
            self.model.workers
        )
        avg_inventory = (
            sum(f.inventory_ratio for f in self.model.factories if not f.bankrupt) / 3
        )

        crisis_level = sum([unemployment > 0.2, avg_inventory > 1.5])

        if crisis_level >= 2:
            self.last_intervention = current_step

            # 财政刺激
            stimulus = 500 * len(self.model.workers)
            for worker in self.model.workers:
                worker.wealth += 500

            # 产业政策
            for factory in self.model.factories:
                if (
                    factory.product_type == ProductType.HEAVY_INDUSTRY
                    and not factory.bankrupt
                ):
                    factory.debt *= 0.7
                    factory.production = int(factory.production * 1.2)

            logging.warning(
                f"=== GOVERNMENT INTERVENTION === "
                f"Unemployment: {unemployment:.1%}, Inventory: {avg_inventory:.1f}, "
                f"Stimulus: {stimulus:.1f}"
            )

    def step(self):
        self.intervene()


class Market:
    def __init__(self, model):
        self.model = model
        self.prices = {
            ProductType.FOOD: 10,
            ProductType.LIGHT_INDUSTRY: 50,
            ProductType.HEAVY_INDUSTRY: 100,
        }
        self.demand = {pt: 0 for pt in ProductType}

    def update_prices(self):
        for product in ProductType:
            supply = sum(
                f.inventory
                for f in self.model.factories
                if f.product_type == product and not f.bankrupt
            )
            demand = self.demand[product] * 2  # 放大需求影响

            if supply == 0:
                self.prices[product] *= 1.5
            else:
                balance = demand / (supply + 1)
                self.prices[product] *= 0.8 + 0.4 * balance

            # 价格下限
            min_price = {
                ProductType.FOOD: 5,
                ProductType.LIGHT_INDUSTRY: 20,
                ProductType.HEAVY_INDUSTRY: 50,
            }[product]
            self.prices[product] = max(self.prices[product], min_price)

        logging.info(
            f"Market prices - Food: {self.prices[ProductType.FOOD]:.1f}, "
            f"Light: {self.prices[ProductType.LIGHT_INDUSTRY]:.1f}, "
            f"Heavy: {self.prices[ProductType.HEAVY_INDUSTRY]:.1f}"
        )

    def sell(self, product, quantity):
        self.demand[product] += quantity


class CrisisModel(mesa.Model):
    def __init__(self, N=5, width=20, height=20):
        super().__init__()
        self.num_workers = N
        # self.grid = mesa.space.MultiGrid(width, height, True)
        self.position_set = set()
        self.schedule = mesa.time.SimultaneousActivation(self)
        self.current_step = 0

        logging.info("===== Initializing Economic Simulation =====")

        # 创建工厂
        products = list(ProductType)
        self.factories = []
        for i, pt in enumerate(products):
            logging.info(f"Creating factory {i} for product {pt.name}")
            pos = ((i + 1) * 5, 10)
            self.position_set.add(pos)
            factory = Factory(i, self, pos, pt)
            # self.grid.place_agent(factory, pos)
            self.schedule.add(factory)
            self.factories.append(factory)
            logging.info(f"Created {pt.name} factory at {pos}")
        self.workers = list(range(self.num_workers))
        # 创建工人
        for i in range(self.num_workers):
            while True:
                temp = (self.random.randrange(width), self.random.randrange(height))
                if temp not in self.position_set:
                    pos = temp
                    self.position_set.add(pos)
                    break
            worker = Worker(i + 10, self, pos)
            self.workers[i] = worker
            logging.info(f"Creating worker {i} at {pos}")
            # self.grid.place_agent(worker, pos)
            self.schedule.add(worker)

        # 创建其他机构
        # self.bank = Bank(10000, self)
        # self.schedule.add(self.bank)
        self.gov = Government(10001, self)
        self.schedule.add(self.gov)
        self.market = Market(self)

        # 数据收集
        # self.datacollector = mesa.DataCollector(
        #     model_reporters={
        #         "Unemployment": lambda m: sum(
        #             1
        #             for a in m.schedule.agents
        #             if isinstance(a, worker) and not a.employed
        #         )
        #         / m.num_workers,
        #         "Avg_Inventory": lambda m: sum(
        #             f.inventory_ratio for f in m.factories if not f.bankrupt
        #         )
        #         / max(1, len([f for f in m.factories if not f.bankrupt])),
        #         "Bank_Reserves": lambda m: m.bank.reserves,
        #         "Avg_Price": lambda m: sum(m.market.prices.values()) / 3,
        #         "GDP": lambda m: sum(
        #             f.production * (3 - f.product_type.value)
        #             for f in m.factories
        #             if not f.bankrupt
        #         ),
        #     },
        #     agent_reporters={
        #         "Wealth": lambda a: a.wealth if isinstance(a, worker) else None,
        #         "Debt": lambda a: a.debt if isinstance(a, Factory) else None,
        #     },
        # )

    def step(self):
        self.current_step += 1
        logging.info(f"\n=== Day {self.current_step} ===")

        # 重置市场需求
        self.market.demand = {pt: 0 for pt in ProductType}

        # 运行所有agent的step
        self.schedule.step()

        # 更新工厂销售数据
        for factory in self.factories:
            if not factory.bankrupt:
                sold = min(factory.inventory, self.market.demand[factory.product_type])
                factory.inventory -= sold
                factory.last_month_sales = sold

        # 更新市场价格
        self.market.update_prices()

        # 收集数据
        # self.datacollector.collect(self)

        # 打印经济摘要
        self.print_summary()

    def print_summary(self):
        unemployment = (
            sum(
                1
                for a in self.schedule.agents
                if isinstance(a, Worker) and not a.employed
            )
            / self.num_workers
        )
        avg_inventory = (
            sum(f.inventory_ratio for f in self.factories if not f.bankrupt) / 3
        )
        bank_health = self.bank.reserves / 100000
        gdp = sum(
            f.production * (3 - f.product_type.value)
            for f in self.factories
            if not f.bankrupt
        )

        logging.info(
            f"Economy Summary:\n"
            f"- Unemployment: {unemployment:.1%}\n"
            f"- Avg Inventory: {avg_inventory:.1f} months\n"
            f"- Bank Reserves: {self.bank.reserves:.1f} ({bank_health:.1%})\n"
            f"- GDP: {gdp:.1f}\n"
            f"- Bankrupt Factories: {sum(1 for f in self.factories if f.bankrupt)}/3"
        )

        # 检测经济周期阶段
        if unemployment < 0.05:
            phase = "Expansion"
        elif unemployment < 0.1:
            phase = "Peak"
        elif unemployment < 0.2:
            phase = "Recession"
        else:
            phase = "Depression"

        logging.info(f"Economic Phase: {phase}")


def run_simulation():
    model = CrisisModel(N=5)
    for i in range(365):  # 模拟1年
        model.step()

    # 保存数据
    df = pd.DataFrame(model.datacollector.model_vars)
    df.to_csv("economic_simulation_results.csv")
    logging.info("Simulation completed. Data saved to CSV.")


if __name__ == "__main__":
    run_simulation()
