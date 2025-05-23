模拟证明思路：

我们借助Mesa有一张可视化的小地图来模拟一个小的封闭经济体。

我们有一个24刻时钟来控制agent行为。
## 想要观察的点？
1:经济危机的产生：

表现为工厂产品大量积压，工人工资降低直到失业，没有工资来购买产品(相对过剩)，
工厂大量倒闭，银行无法从工厂收回债务偿还工人债务导致破产。

2:要观察到经济危机的周期性出现

3：观察到每次经济危机的二重性，即每次经济危机给经济带来重创，但同时也重塑生产关系，引导宏观调控解决原有的矛盾。

## 成员：
### 工人 (Consumers)：

#### 属性:

有一个家的位置： 工人每天如果上班，会从家的位置走到工厂开始上班。


生活幸福度： 动态变化，每天更新，由每天的工作时间和财产数额决定。（非线性，如工作时间超过10小时后幸福度下降更快，财产数额接近0时幸福度下降更快）


工作的工厂： 在唯一的一个工厂工作或者不工作，每30天更新一次。

每天工作的时间： 如果不工作就设为0, 否则在6-16小时之间。动态变化，工人财富（相对财富）越多，工作时间越少。

每小时工资： 由工作的工厂决定。


#### 行为逻辑：

工人可以在工厂每天工作6-16小时（为了方便我们把工作时间定义为连续的工作，不可以间断），

工人也可以在家躺平玩手机刷视频。

每次在工厂的聘期为30天。如果不工作，也得等30天的聘期过去。但是每天的工作时间可以动态变化。


收入可以从市场中购买产品。


### 工厂 （资本家）：我们这里把工厂和资本家视为一类存在。

我们为了简化模型，只有一个工厂。他生产抽象的产品（product）

#### 工厂的属性：

1 在地图上存在一个位置

2 工厂有对应的每小时生产工资。

3 每种工厂根据市场需求调整生产量，同时因为市场调节的滞后性，工厂只会参考上个月的市场需求量。

初始化：初始条件为正好符合实际需求，交由市场进行调节。

如果市场需求低，生产者降低产量；如果需求旺盛，增加产量。

工厂如何增加产量？表现为雇佣工人的位置增多。同时增加部分工资以吸引劳动力。

但是如果库存发生积压，工人工资会相应地降低。



市场： 从工厂接受产品，售卖价格由工人工资，供需比例共同决定，同时由于产品本身销售的成本，我们禁止产品价格低于某一常数，也禁止产品价格低于工人工资。 

，我们或许可以忽略掉产品运送到市场的过程。 
`
市场将产品售卖给工人, 工人在不工作的时间从家里走到市场购买商品。


### 希望观察到的现象
初期，市场价格相对稳定。中期，生产者的产量持续增长，将会出现总供给超过总需求的情况，导致市场价格下降。

后期，市场价格持续下降，生产者的利润空间被压缩，导致工厂裁员，降低工资，从而导致总供给减少。然而，如果这一调整不足以恢复供需平衡，可能会形成一个恶性循环，即更多的生产者因利润减少而进一步削减生产或倒闭，最终导致经济衰退或危机。其他：随着时间的发展，消费者的储蓄可能会逐渐增加，因为他们的一部分收入没有用于消费而是被储存起来。同时，生产者的库存可能会累积，尤其是在市场需求不足的情况下。这种情况表明了生产和消费之间的脱节。

然后采取一些宏观调控方式，重塑供需关系，重新经历市场复苏，繁荣，再次进入危机的过程。
