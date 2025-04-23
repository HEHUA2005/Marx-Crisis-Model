import mesa
import numpy as np
import logging
import pandas as pd
from mesa import DataCollector


def get_distance(pos1, pos2):
    """计算两个位置之间的曼哈顿距离"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


class Worker(mesa.Agent):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.home_pos = pos
        self.wealth = 10000  # 初始有10000块积蓄
        self.employed = False  # 初始没有工作
        self.factory = None
        self.work_duration = 8  # 工作时长
        self.health = 100  # 健康值
        self.happiness = 50  # 幸福值
        self.consumption_cycle = 1
        self.last_consumption = 0

    def __str__(self):
        return f"Worker {self.unique_id}: Wealth={self.wealth:.1f}, Employed={self.employed}"

    def choose_factory(self):
        if self.model.schedule.steps % 30 != 0 and self.employed:
            return
        if self.wealth > 10000:
            logging.info(
                f"Worker {self.unique_id} is too rich to work, wealth: {self.wealth:.1f}"
            )
            if self.employed:
                self.employed = False
                self.factory.workers.remove(self)
                self.factory = None
            return
        if not self.model.factory.hiring:
            if self.employed and self.factory.bankrupt:
                self.employed = False
                logging.info(
                    f"Worker {self.unique_id} lost job due to factory bankruptcy"
                )
                return
            if not self.employed:
                logging.info(f"Worker {self.unique_id} found no hiring factories")
                return
            return

    def work(self):
        if not self.employed:
            return

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
        # 花钱购买产品 消费会增加Happiness
        if not self.living:
            return
        price = self.model.market.prices
        if self.wealth - price >= 0:
            self.wealth -= price
            self.model.market.sell(1)
            self.happiness += 5
            self.last_consumption = self.model.schedule.steps
            logging.debug(
                f"Worker {self.unique_id} bought product at price {price:.1f}"
            )
        else:
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
        self.target_production = 0
        self.daily_production = 0
        self.inventory = 0
        self.workers = []
        self.hiring = True
        self.debt = 0
        self.bankrupt = False
        self.wealth = 0
        self.job_offer = 5

    def __str__(self):
        status = f"Production={self.production}, Workers={len(self.workers)}"
        return f"Factory status:({self.product_type.name}): {status}"

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

        if sold_ratio > 0.8:
            self.wage *= 1.05
        elif sold_ratio < 0.5:
            self.wage *= 0.95
        self.wage = max(5, min(50, self.wage))

        # 债务处理
        # if self.debt > self.production * 10:
        #     self.debt *= 1.05  # 债务增长
        #     if np.random.random() < 0.1:  # 10%概率获得银行贷款
        #         self.debt *= 0.9

        logging.info(
            f"Factory {self.unique_id} adjusted production: {old_production} -> {self.production} "
            f"(Sold ratio: {sold_ratio:.2f}, Wage: {self.wage:.1f})"
        )

    def produce(self):
        self.daily_production = 0
        for worker in self.workers:
            self.daily_production += 1 * worker.work_duration
            self.wealth += self.wage
        self.inventory += self.production
        self.last_month_sales = max(0, self.last_month_sales)

    def step(self):
        self.adjust_production()
        self.check_bankruptcy()
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
        self.monthly_sales = 50
        self.monthly_demand = 50
        self.last_update = 0
        self.update_interval = 30  # 每30天更新一次市场价格
        self.last_month_sales = 50
        self.last_month_demand = 50

    def update_prices(self):
        supply = max(0, self.model.factory.inventory)
        self.monthly_demand = self.monthly_sales * 1.2  # 放大需求影响

        if supply == 0:
            self.prices *= 2
        else:
            balance = self.monthly_demand / (supply + 1)
            self.prices *= 0.8 + 0.4 * balance

        # 价格下限

        logging.info(f"Market prices : {self.prices}")

    def sell(self, quantity):
        self.model.factories.inventory -= quantity
        self.monthly_sales += quantity

    def step(self):
        if self.model.schedule.steps % self.update_interval != 0:
            return
        self.update_prices()
        self.last_month_sales = self.monthly_sales.copy()
        self.last_month_demand = self.monthly_demand.copy()
        self.monthly_sales = 0
        self.monthly_demand = 0
