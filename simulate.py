import mesa
import numpy as np
import logging
from enum import Enum
from datetime import datetime
import pandas as pd
from mesa import DataCollector
from Agents import (
    Worker,
    Factory,
    Government,
    Market,
)

# 设置日志系统
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("economic_simulation.log"), logging.StreamHandler()],
)


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
        pos = (10, 10)
        self.factory = Factory(0, self, pos)

        logging.info(f"Creating factory in {pos}")
        self.position_set.add(pos)

        self.schedule.add(self.factory)

        # 创建工人
        self.workers = list(range(self.num_workers))
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
            self.schedule.add(worker)

        # Create a government agent and a market
        self.gov = Government(10001, self)
        self.schedule.add(self.gov)
        self.market = Market(10000, self)
        self.schedule.add(self.market)

    def step(self):
        self.current_step += 1
        logging.info(f"\n=== Day {self.current_step} ===")

        # 运行所有agent的step
        self.schedule.step()

        # 更新工厂销售数据
        for factory in self.factories:
            if not factory.bankrupt:
                sold = min(factory.inventory, self.market.demand[factory.product_type])
                factory.inventory -= sold
                factory.last_month_sales = sold

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
        inventory = sum(self.factory.inventory)
        gdp = sum(
            f.production * (3 - f.product_type.value)
            for f in self.factories
            if not f.bankrupt
        )

        logging.info(
            f"Economy Summary:\n"
            f"- Unemployment: {unemployment:.1%}\n"
            f"- Inventory: {inventory:.1f} months\n"
            f"- GDP: {gdp:.1f}\n"
            f"- Bankrupt Factories: {sum(1 for f in self.factories if f.bankrupt)}/3"
        )

        # 检测经济周期阶段
        if unemployment < 0.05:
            phase = "Expansion"
        elif unemployment < 0.2:
            phase = "Recession"
        else:
            phase = "An economic crisis occurred!"

        logging.info(f"Economic Phase: {phase}")


def run_simulation():
    model = CrisisModel(N=20)
    for i in range(60):  # 模拟1年
        model.step()


if __name__ == "__main__":
    run_simulation()
