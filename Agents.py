import mesa
import numpy as np
import logging
import pandas as pd
from mesa import DataCollector
import random


def get_distance(pos1, pos2):
    """计算两个位置之间的曼哈顿距离"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


class Worker(mesa.Agent):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.home_pos = pos
        self.wealth = 10000  # 初始有10000块积蓄
        self.relative_wealth = 500
        self.employed = False  # 初始没有工作
        self.work_duration = 8  # 工作时长
        self.factory = None
        self.happiness = 50  # 幸福值
        self.consumption_cycle = 1
        self.last_consumption = 0

    def __str__(self):
        return f"Worker {self.unique_id}: Wealth={self.wealth:.1f}, Employed={self.employed}"

    def choose_factory(self):
        if self.model.schedule.steps % 30 != 0 and self.employed:
            return

        if self.wealth > 15000:
            logging.info(
                f"Worker {self.unique_id} is too rich to work, wealth: {self.wealth:.1f}"
            )
            if self.employed:
                self.employed = False
                self.factory.workers.remove(self)
            return
        if not self.model.factory.hiring:
            if self.employed:
                logging.info(f"Worker {self.unique_id} keep job.")
                return
            if not self.employed:
                logging.info(f"Worker {self.unique_id} found no hiring factories")
                return
            return
        self.employed = True
        self.factory = self.model.factory
        self.factory.workers.append(self)
        self.factory.job_offer -= 1
        self.factory.hiring = self.factory.job_offer - len(self.factory.workers) > 0

    def work(self):
        if not self.employed:
            self.happiness += 3
            return
        self.happiness -= 1
        self.work_duration = 8
        self.work_duration = np.clip(self.work_duration, 4, 16)

        # 获得收入
        income = self.work_duration * self.factory.wage
        self.wealth += income

        # 健康变化
        if self.work_duration > 12:
            self.happiness -= 3
        elif self.work_duration < 8:
            self.happiness += 1

    def consume(self):
        # 花钱购买产品 消费会增加Happiness
        price = self.model.market.prices
        self.relative_wealth = self.wealth / price
        need = random.randint(1, 4)
        if self.wealth - price * need >= 0 and self.model.factory.inventory > need:
            self.wealth -= price
            self.model.market.sell(1)
            self.happiness += 5
            self.last_consumption = self.model.schedule.steps
            logging.debug(
                f"Worker {self.unique_id} bought product at price {price:.1f}"
            )
        elif self.model.factory.inventory < need:
            self.happiness -= 1
            logging.debug(f"Worker {self.unique_id} cannot find product in market")
            return
        else:
            self.happiness -= 1
            logging.debug(
                f"Worker {self.unique_id} cannot afford product at price {price:.1f}"
            )
            return

    def step(self):
        self.choose_factory()
        self.work()
        self.consume()


class Factory(mesa.Agent):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.pos = pos
        self.wage = 15
        self.target_monthly_production = 0
        self.daily_production = 0
        self.last_daily_production = 0
        self.monthly_production = 0
        self.last_monthly_production = 0
        self.inventory = 0
        self.workers = []
        self.hiring = True
        self.debt = 0
        self.wealth = 0
        self.job_offer = 30

    def __str__(self):
        status = f"Production={self.production}, Workers={len(self.workers)}"
        return f"Factory status:({self.product_type.name}): {status}"

    @property
    def inventory_ratio(self):
        if self.monthly_production == 0:
            return 10000
        return self.inventory / self.monthly_production

    def adjust_production(self):
        if self.model.schedule.steps % 30 != 0:
            return
        self.last_monthly_production = self.monthly_production
        self.monthly_production = 0
        sold_ratio = min(
            1,
            self.model.market.last_month_sales / (self.last_monthly_production + 1e-5),
        )
        adjustment = 0.5 + 0.6 * sold_ratio

        if adjustment > 1:
            self.job_offer += 1
        elif adjustment < 0.9:
            self.job_offer -= 1
        logging.info(
            f"Factory {self.unique_id} adjusted production ratio: {adjustment}  "
        )

    def produce(self):
        self.last_daily_production = self.daily_production
        self.daily_production = 0
        for worker in self.workers:
            self.daily_production += int(0.33 * worker.work_duration)
            self.wealth -= self.wage
        self.inventory += self.daily_production
        self.monthly_production += self.daily_production

    def step(self):
        self.adjust_production()
        self.produce()


class Government(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.last_intervention = 0

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


class Market(mesa.Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id=unique_id, model=model)
        self.model = model
        self.prices = 20
        self.min_price = 1
        self.monthly_sales = 0
        self.daily_sales = 0
        self.last_month_sales = 3750
        self.last_daily_sales = 125

    def update_prices(self):
        supply = max(0, self.model.factory.inventory)

        if supply == 0:
            self.prices *= 2
        else:
            balance = self.last_daily_sales / supply
            self.prices *= 0.8 + 0.2 * balance
        self.prices = max(self.min_price, self.prices)

        logging.info(f"Market prices : {self.prices}")

    def sell(self, quantity):
        self.model.factory.inventory -= quantity
        self.daily_sales += quantity
        self.monthly_sales += quantity

    def step(self):
        self.update_prices()
        if self.model.steps % 30 == 0:
            self.last_month_sales = self.monthly_sales
            self.monthly_sales = 0
