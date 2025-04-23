import mesa
import logging
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
        self.market.step()
        self.factory.step()
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
        logging.info(
            f"Economy Summary:\n"
            f"- Unemployment: {unemployment:.1%}\n"
            f"- Inventory: {inventory:.1f} \n"
            f"- Daily GDP: {daily_gdp:.1f}\n"
            f"- Daily production: {f.daily_production:.1f}\n"
            f"- Average workers' Wealth: {average_worker_wealth:.1f}\n"
            f"- Factory wealth: {f.wealth:.1f}\n"
            f"- Market daily sales: {self.market.daily_sales:.1f}\n"
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
    model = CrisisModel(N=50)
    for i in range(100):  # 模拟天数
        model.step()


if __name__ == "__main__":
    run_simulation()
