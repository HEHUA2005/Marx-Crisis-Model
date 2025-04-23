import mesa
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
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
    handlers=[
        logging.FileHandler("economic_simulation.log", mode="w"),
        logging.StreamHandler(),
    ],
)


class CrisisModel(mesa.Model):
    def __init__(self, N=5, width=20, height=20):
        super().__init__()
        # 用来画图
        self.time_steps = []
        self.unemployment_rates = []
        self.inventories = []
        self.daily_gdps = []
        self.daily_productions = []
        self.avg_worker_wealths = []
        self.factory_wealths = []
        self.market_sales = []
        self.avg_happiness = []
        #

        self.steps = 0
        self.num_workers = N
        # self.grid = mesa.space.MultiGrid(width, height, True)
        self.position_set = set()
        self.schedule = mesa.time.SimultaneousActivation(self)

        logging.info("===== Initializing Economic Simulation =====")

        # 创建工厂
        pos = (10, 10)
        self.factory = Factory(0, self, pos)

        logging.info(f"Creating factory in {pos}")
        self.position_set.add(pos)

        # 创建工人
        self.workers = list(range(self.num_workers))
        for i in range(self.num_workers):
            while True:
                temp = (self.random.randrange(width), self.random.randrange(height))
                if temp not in self.position_set:
                    pos = temp
                    self.position_set.add(pos)
                    break
            worker = Worker(i + 1, self, pos)
            self.workers[i] = worker
            logging.info(f"Creating worker {i} at {pos}")
            self.schedule.add(worker)

        # Create a government agent and a market
        self.gov = Government(10001, self)
        self.market = Market(10000, self)

    def step(self):
        self.steps += 1

        logging.info(f"\n=== Day {self.steps} ===")

        # 运行所有worker的step
        self.schedule.step()
        self.factory.step()
        self.market.step()
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
        inventory = self.factory.inventory
        f = self.factory
        daily_gdp = f.daily_production * self.market.prices
        average_worker_wealth = sum(
            [w.wealth for w in self.schedule.agents if isinstance(w, Worker)]
        ) / len([w for w in self.schedule.agents if isinstance(w, Worker)])
        avg_happiness = sum(
            [w.happiness for w in self.schedule.agents if isinstance(w, Worker)]
        ) / len([w for w in self.schedule.agents if isinstance(w, Worker)])
        logging.info(
            f"Economy Summary:\n"
            f"- Unemployment: {unemployment:.1%}\n"
            f"- Inventory: {inventory:.1f} \n"
            f"- Daily GDP: {daily_gdp:.1f}\n"
            f"- Daily production: {f.daily_production:.1f}\n"
            f"- Average workers' Wealth: {average_worker_wealth:.1f}\n"
            f"- Factory wealth: {f.wealth:.1f}\n"
            f"- Market daily sales: {self.market.last_daily_sales:.1f}\n"
            f"- Average happiness: {avg_happiness:.1f}\n"
        )
        self.time_steps.append(self.steps)
        self.unemployment_rates.append(unemployment)
        self.inventories.append(inventory)
        self.daily_gdps.append(daily_gdp)
        self.daily_productions.append(f.daily_production)
        self.avg_worker_wealths.append(average_worker_wealth)
        self.factory_wealths.append(f.wealth)
        self.market_sales.append(self.market.last_daily_sales)
        self.avg_happiness.append(avg_happiness)
        # 检测经济周期阶段
        if unemployment < 0.05:
            phase = "Expansion"
        elif unemployment < 0.2:
            phase = "Recession"
        else:
            phase = "An economic crisis occurred!"

        logging.info(f"Economic Phase: {phase}")

    def plot_statistics(self):
        # 绘制数据图
        fig, axs = plt.subplots(4, 2, figsize=(15, 12))
        fig.delaxes(axs[3, 1])  # 删除最后一个多余的子图

        metrics = [
            ("Unemployment Rate", self.unemployment_rates, "百分比"),
            ("Inventory", self.inventories, "单位"),
            ("Daily GDP", self.daily_gdps, "货币单位"),
            ("Daily Production", self.daily_productions, "单位"),
            ("Worker Wealth", self.avg_worker_wealths, "货币单位"),
            ("Factory Wealth", self.factory_wealths, "货币单位"),
            ("Market Sales", self.market_sales, ""),
            ("Average Happiness", self.avg_happiness, "points"),
        ]

        for idx, (title, data, unit) in enumerate(metrics):
            row = idx // 2
            col = idx % 2
            axs[row, col].plot(self.time_steps, data, "b-")
            axs[row, col].set_title(title)
            axs[row, col].set_xlabel("Time Step")
            axs[row, col].set_ylabel(unit)
            axs[row, col].grid(True)

        plt.tight_layout()
        plt.savefig("economy_stats.png")  # 保存图表
        plt.show()


def run_simulation():
    model = CrisisModel(N=50)
    for i in range(100):  # 模拟天数
        model.step()
    model.plot_statistics()  # 绘制统计图


if __name__ == "__main__":
    logging.info("Starting simulation...")
    run_simulation()
