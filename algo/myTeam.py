# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import capture
import random, time, util
from game import Directions
import game
from util import nearestPoint
import math


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
    """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########



#隧道会存储地图上所有隧道的位置，隧道意味着
# #只有一条路可以离开
tunnels = []

#隧道将存储地图上所有隧道位置，但以
#敌方边界为墙来寻找隧道。
defensiveTunnels = []

# 存储地图的墙壁

# tunnels will store all tunnel positions of the map, the tunnel means
# only one way to leave
#tunnels（隧道）储存地图中所有隧道坐标，所谓隧道坐标即只有一种离开方式，至少有两种移动方式不能选择
tunnels = []

# tunnels will store all tunnel positions of the map, but regarding the
# boundary as wall to find tunnels.
#defensiveTunnels(防守隧道)储存本方地图上的隧道坐标
defensiveTunnels = []

# store the walls of the map
#walls储存所有墙所在的坐标

walls = []

"""
getAllTunnels 将以列表形式返回所有隧道，
它使用 while 循环逐级查找隧道，直到地图中没有更多隧道为止
"""
def getAllTunnels(legalPositions):
    """
    getAllTunnels will return the all tunnels as a list, it uses a while loop 
    to find a tunnel level by level, stop until no more tunnels in the map
    getAllTunnels返回列表，储存所有通道坐标
    实现方法：
    使用while循环分层遍历
    调用getMoreTunnels实现下一层搜索
    直到搜索完所有隧道点
    """
    tunnels = []
    while len(tunnels) != len(getMoreTunnels(legalPositions, tunnels)):
        tunnels = getMoreTunnels(legalPositions, tunnels)
    return tunnels


"""
getMoreTunnels 是查找下一层隧道的函数
"""
def getMoreTunnels(legalPositions, tunnels):
    """
    getMoreTunnels is the function to find the next level's tunnel
    getMoreTunnels返回列表，实现隧道坐标的分层搜索
    实现方法：
    遍历所有LegalPositions(合法坐标，agnet能存在的位置)
    通过getSuccsorNum分别查找遍历坐标在tunnels中的可移动路线数量neighborTunnelsNum
    和在legalPositions中的可移动路线数量succsorsNum
    若succsorsNum - neighborTunnelsNum == 1且遍历的坐标不在tunnels中
    则加入tunnels
    """
    newTunnels = tunnels
    for i in legalPositions:
        neighborTunnelsNum = getSuccsorsNum(i, tunnels)
        succsorsNum = getSuccsorsNum(i, legalPositions)
        if succsorsNum - neighborTunnelsNum == 1 and i not in tunnels:
            newTunnels.append(i)
    return newTunnels


"""
getSuccsorsNum是记录下一步可以走的数量
"""
def getSuccsorsNum(pos, legalPositions):
    """
    getMoreTunnels is the function to find the next level's tunnel
    getMoreTunnels返回一个数，表示在传入的legalPositions中的可移动路线数量
    实现方法：
    分别判断上下左右的临近坐标是否在legalPositions中
    若在，则num+1
    """
    num = 0
    x, y = pos
    if (x + 1, y) in legalPositions:
        num += 1
    if (x - 1, y) in legalPositions:
        num += 1
    if (x, y + 1) in legalPositions:
        num += 1
    if (x, y - 1) in legalPositions:
        num += 1
    return num


"""
getSuccsorsPos 将返回所有位置的合法邻居位置
"""


def getSuccsorsPos(pos, legalPositions):
    """
    getSuccsorsPos will return all position's legal neighbor positions
    getSuccsorsPos返回一个列表，储存在传入的legalPositions中的临近坐标
    实现方法：
    分别判断上下左右的临近坐标是否在legalPositions中
    若在，则加入列表   
    """
    succsorsPos = []
    x, y = pos
    if (x + 1, y) in legalPositions:
        succsorsPos.append((x + 1, y))
    if (x - 1, y) in legalPositions:
        succsorsPos.append((x - 1, y))
    if (x, y + 1) in legalPositions:
        succsorsPos.append((x, y + 1))
    if (x, y - 1) in legalPositions:
        succsorsPos.append((x, y - 1))
    return succsorsPos


"""
给定当前位置和一个动作，nextPos 将返回下一个位置
"""
def nextPos(pos, action):
    """
    given current position and an action, nextPos will return the next position
    nextPos返回一个坐标，表示在action操作下pos的变化
    实现方法：
    判断action类型，根据类型做出相应改动
    """
    x, y = pos
    if action == Directions.NORTH:
        return (x, y + 1)
    if action == Directions.SOUTH:
        return (x, y - 1)
    if action == Directions.EAST:
        return (x + 1, y)
    if action == Directions.WEST:
        return (x - 1, y)
    return pos


"""
manhattanDist：输入两个点，返回这两点之间的曼哈顿距离
"""


def manhattanDist(pos1, pos2):
    """
    manhattanDist: input two points, return the mahattan distance between 
    these two points
    manhattanDist返回一个数，表示输入的pos1和pos2之间的曼哈顿距离
    """
    x1, y1 = pos1
    x2, y2 = pos2
    return abs(x2 - x1) + abs(y2 - y1)


"""
getTunnelEntry：给定一个位置，如果位置在隧道中，它将返回此隧道的入口位置
"""


def getTunnelEntry(pos, tunnels, legalPositions):
    """
    getTunnelEntry: given a position, if position in tunnels, it will return
    the entry position of this tunnel
    getTunnelEntry返回一个坐标，表示pos所在隧道的入口坐标
    实现方法：
    先判断pos是否在tunnels中，若不在，返回None
    若在，通过getATunnels获得pos所在的当前隧道的所有坐标aTunnel
    遍历aTunnel，通过getPossibleEntry判断是否是入口
    """
    if pos not in tunnels:
        return None
    aTunnel = getATunnel(pos, tunnels)
    for i in aTunnel:
        possibleEntry = getPossibleEntry(i, tunnels, legalPositions)
        if possibleEntry != None:
            return possibleEntry


"""
getPossibleEntry：此辅助函数用于 getTunnelEntry 查找下一个邻居位置是否是隧道入口
"""


def getPossibleEntry(pos, tunnels, legalPositions):
    """
    getPossibleEntry: this assisted funtion used in getTunnelEntry to
    find if next neighbor position is tunnel entry
    getPossibleEntry返回一个坐标，表示一条隧道的入口
    实现方法：
    判断当前pos的临近坐标是否满足：
    （1）在legalPositions中
    （2）不在tunnels中
    若满足，则返回这个临近坐标
    """
    x, y = pos
    if (x + 1, y) in legalPositions and (x + 1, y) not in tunnels:
        return (x + 1, y)
    if (x - 1, y) in legalPositions and (x - 1, y) not in tunnels:
        return (x - 1, y)
    if (x, y + 1) in legalPositions and (x, y + 1) not in tunnels:
        return (x, y + 1)
    if (x, y - 1) in legalPositions and (x, y - 1) not in tunnels:
        return (x, y - 1)
    return None


"""
getATunnel：输入一个位置和隧道，该函数将返回该位置所属的隧道
"""


def getATunnel(pos, tunnels):
    """
    getATunnel: input a position and tunnels, this function will return a tunnel
    that this position belongs to
    getATunnel返回一个列表，存储当前pos所在隧道的所有坐标
    实现方法：
    先判断pos在不在tunnels中,
    若在，构建一个FIFO的队列bfs_queue和一个空列表closed
    将当前pos加入bfs_queue
    若bfs_queue非空，则
        取出一个坐标currPos，若currPos不在closed中
        加入closed，并寻找currPos的临近坐标
        遍历临近坐标，若临近坐标不在closed中，
        加入bfs_queue
    """
    if pos not in tunnels:
        return None
    bfs_queue = util.Queue()
    closed = []
    bfs_queue.push(pos)
    while not bfs_queue.isEmpty():
        currPos = bfs_queue.pop()
        if currPos not in closed:
            closed.append(currPos)
            succssorsPos = getSuccsorsPos(currPos, tunnels)
            for i in succssorsPos:
                if i not in closed:
                    bfs_queue.push(i)
    return closed


"""
类节点：用于 UCT 流程。节点是不同的游戏状态，它具有一些功能：
addChild，添加子节点。
findParnt：查找此节点的父节点。
chooseChild：选择具有最高 UCT 值的子节点
"""


class Node:
    def __init__(self, value, id=0):
        (gameState, t, n) = value
        self.id = id
        self.children = []
        self.value = (gameState, float(t), float(n))
        self.isLeaf = True

    def addChild(self, child):
        self.children.append(child)

    def chooseChild(self):
        _, _, pn = self.value
        maxUCB = -999999
        bestChild = None
        for i in self.children:
            _, t, n = i.value
            if n == 0:
                return i
            UCB = t + 1.96 * math.sqrt(math.log(pn) / n)
            if maxUCB < UCB:
                maxUCB = UCB
                bestChild = i
        return bestChild

    def findParent(self, node):
        for i in self.children:
            if i == node:
                return self
            else:
                possibleParent = i.findParent(node)
                if possibleParent != None:
                    return possibleParent

    def __str__(self):
        (_, t, n) = self.value
        id = self.id
        return "Node " + str(id) + ", t = " + str(t) + ", n = " + str(n)


"""
类树：用于 UCT 过程中，用于存储节点，它具有以下一些功能：
insert: 给定父节点和子节点，将此子节点添加到树中
getParent: 使用 findParent 函数返回父节点
backPropagate: 标准UCT反向传播过程，进行更新工作
select: 迭代查找具有最大 UCT 值的子项

"""


class Tree:
    def __init__(self, root):
        self.count = 1
        self.tree = root
        self.leaf = [root.value[0]]

    def insert(self, parent, child):
        id = self.count
        self.count += 1
        child.id = id
        parent.addChild(child)
        if parent.value[0] in self.leaf:
            self.leaf.remove(parent.value[0])
        parent.isLeaf = False
        self.leaf.append(child.value[0])

    def getParent(self, node):
        if node == self.tree:
            return None
        return self.tree.findParent(node)

    def backPropagate(self, r, node):
        (gameState, t, n) = node.value
        node.value = (gameState, t + r, n + 1)
        parent = self.getParent(node)
        if parent != None:
            self.backPropagate(r, parent)

    def select(self, node=None):
        if node == None:
            node = self.tree
        if not node.isLeaf:
            nextNode = node.chooseChild()
            return self.select(nextNode)
        else:
            return node

    """
该类生成入侵者位置的信念，该类中应用了 HMM 模型
  """


class ParticleFilter:

    def __init__(self, agent, gameState):

        self.start = gameState.getInitialAgentPosition(agent.index)
        self.agent = agent
        self.midWidth = gameState.data.layout.width / 2
        self.legalPositions = [p for p in gameState.getWalls().asList(False)]
        self.enemies = self.agent.getOpponents(gameState)
        self.beliefs = {}
        for enemy in self.enemies:
            self.beliefs[enemy] = util.Counter()
            self.beliefs[enemy][gameState.getInitialAgentPosition(enemy)] = 1.0
            self.beliefs[enemy].normalize()

    # 该函数更新入侵者的分布
    # 均匀分布的位置
    def elapseTime(self):

        for enemy in self.enemies:
            dist = util.Counter()

            for p in self.legalPositions:
                newDist = util.Counter()

                allPositions = [(p[0] + i, p[1] + j) for i in [-1, 0, 1] for j in [-1, 0, 1] if
                                not (abs(i) == 1 and abs(j) == 1)]

                for q in self.legalPositions:
                    if q in allPositions:
                        newDist[q] = 1.0
                newDist.normalize()

                for pos, probability in newDist.items():
                    dist[pos] = dist[pos] + self.beliefs[self.enemy][pos] * probability

            dist.normalize()
            self.beliefs[enemy] = dist

    # 该函数使用噪声距离来缩小入侵者位置的概率范围
    def observe(self, agent, gameState):

        myPos = gameState.getAgentPosition(agent.index)
        noisyDistance = gameState.getAgentDistances()

        dist = util.Counter()

        for enemy in self.enemies:
            for pos in self.legalPositions:

                trueDistance = util.manhattanDistance(myPos, pos)
                probability = gameState.getDistanceProb(trueDistance, noisyDistance)

                if agent.red:
                    ifPacman = pos[0] < self.midWidth
                else:
                    ifPacman = pos[0] > self.midWidth

                if trueDistance <= 6 or ifPacman != gameState.getAgentState(enemy).isPacman:
                    dist[pos] = 0.0
                else:
                    dist[pos] = self.beliefs[enemy][pos] * probability

            dist.normalize()
            self.beliefs[enemy] = dist

    def getPossiblePosition(self, enemy):

        pos = self.beliefs[enemy].argMax()
        return pos


class ReflexCaptureAgent(CaptureAgent):
    """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

    def registerInitialState(self, gameState):
        """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

        '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.

    '''

        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

        self.changeEntrance = False  # 控制改变入口功能
        self.nextEntrance = None  # 如果需要改变入口，存储下一个入口的位置
        self.carriedDot = 0  # 存储进攻代理携带的点数
        self.tunnelEntry = None  # 如果代理在隧道中，存储此隧道入口
        global walls  # 声明全局类型
        global tunnels  # 声明全局类型
        global openRoad  # 声明全局类型，openRoad 是隧道外的位置
        global legalPositions  # 声明全局类型
        walls = gameState.getWalls().asList()
        if len(tunnels) == 0:
            legalPositions = [p for p in gameState.getWalls().asList(False)]
            tunnels = getAllTunnels(legalPositions)
            openRoad = list(set(legalPositions).difference(set(tunnels)))
        self.capsule = None  # 存储代理将跑向的安全胶囊
        self.nextOpenFood = None  # 存储代理将跑向的开放道路中最接近的安全食物
        self.nextTunnelFood = None  # 存储代理将跑向的隧道中最接近的安全食物
        self.runToBoundary = None  # 存储最接近的边界位置
        self.stuckStep = 0  # 如果我们的代理与对手卡住，计数步数
        self.curLostFood = None  # 存储被入侵者吃掉的食物
        self.ifStuck = False  # 当发现卡住时，此变量变为真，开始计数步数
        self.enemyGuess = ParticleFilter(self, gameState)  # 存储入侵者的猜测位置
        self.invadersGuess = False  # 如果发现入侵者，此变量将变为真
        global defensiveTunnels  # 声明全局类型
        width = gameState.data.layout.width
        legalRed = [p for p in legalPositions if p[0] < width / 2]  # 红色区域中的合法位置
        legalBlue = [p for p in legalPositions if p[0] >= width / 2]  # 蓝色区域中的合法位置
        if len(defensiveTunnels) == 0:
            if self.red:
                defensiveTunnels = getAllTunnels(legalRed)
            else:
                defensiveTunnels = getAllTunnels(legalBlue)

    """
    chooseAction：如果 self.ifStuck 为 True，它将调用 MCTs 函数进行决策。
    否则它将使用评估函数找到最佳操作
    """

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)

        values = [self.evaluate(gameState, a) for a in actions]

        Q = max(values)

        if self.ifStuck:
            return self.simulation(gameState)

        bestActions = [a for a, v in zip(actions, values) if v == Q]

        action = random.choice(bestActions)

        return action

    """
    查找下一个作为网格位置（位置元组）的后继者。
    """

    def getSuccessor(self, gameState, action):

        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    """
    计算特征和特征权重的线性组合
    """

    def evaluate(self, gameState, action):

        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)

        return features * weights

    """
    如果代理在隧道入口处，则将调用此方法来评估此隧道。
    如果此隧道中没有食物，则返回 0。
    否则返回隧道中最近的食物与代理之间的距离
    """

    def ifWasteTunnel(self, gameState, successor):

        curPos = gameState.getAgentState(self.index).getPosition()
        sucPos = successor.getAgentState(self.index).getPosition()
        if curPos not in tunnels and sucPos in tunnels:

            self.tunnelEntry = curPos

            dfs_stack = util.Stack()
            closed = []
            dfs_stack.push((sucPos, 1))

            while not dfs_stack.isEmpty():
                (x, y), length = dfs_stack.pop()
                if self.getFood(gameState)[int(x)][int(y)]:
                    return length

                if (x, y) not in closed:
                    closed.append((x, y))
                    succssorsPos = getSuccsorsPos((x, y), tunnels)
                    for i in succssorsPos:
                        if i not in closed:
                            nextLength = length + 1
                            dfs_stack.push((i, nextLength))
        return 0

    # 使用 BFS 搜索获取隧道中最近的食物
    def getTunnelFood(self, gameState):

        curPos = gameState.getAgentState(self.index).getPosition()
        bfs_queue = util.Queue()
        closed = []
        bfs_queue.push(curPos)

        while not bfs_queue.isEmpty():
            x, y = bfs_queue.pop()
            if self.getFood(gameState)[int(x)][int(y)]:
                return (x, y)

            if (x, y) not in closed:
                closed.append((x, y))
                succssorsPos = getSuccsorsPos((x, y), tunnels)
                for i in succssorsPos:
                    if i not in closed:
                        bfs_queue.push(i)

        return None

    # 获取剩余时间
    def getTimeLeft(self, gameState):
        return gameState.data.timeleft

    # 获得所有合法位置以跳转到中间边界
    def getEntrance(self, gameState):
        width = gameState.data.layout.width
        height = gameState.data.layout.height
        legalPositions = [p for p in gameState.getWalls().asList(False)]
        legalRed = [p for p in legalPositions if p[0] == width / 2 - 1]
        legalBlue = [p for p in legalPositions if p[0] == width / 2]
        redEntrance = []
        blueEntrance = []
        for i in legalRed:
            for j in legalBlue:
                if i[0] + 1 == j[0] and i[1] == j[1]:
                    redEntrance.append(i)
                    blueEntrance.append(j)
        if self.red:
            return redEntrance
        else:
            return blueEntrance

    # MCT 的子方法。使用随机游走模拟 20 步并返回最后一个状态的值，如果被鬼吃掉就会中断
    def OfsRollout(self, gameState):
        counter = 20
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghost = [a for a in enemies if not a.isPacman and a.getPosition() is not None]
        ghostPos = [a.getPosition() for a in ghost]
        curState = gameState
        while counter != 0:
            counter -= 1
            actions = curState.getLegalActions(self.index)
            nextAction = random.choice(actions)
            successor = self.getSuccessor(curState, nextAction)
            myPos = nextPos(curState.getAgentState(self.index).getPosition(), nextAction)
            if myPos in ghostPos:
                return -9999
            curState = successor
        return self.evaluate(curState, 'Stop')

        # 运行 MCT 的功能。循环时间为 0.95 秒

    def simulation(self, gameState):
        (x1, y1) = gameState.getAgentPosition(self.index)
        root = Node((gameState, 0, 0))
        mct = Tree(root)
        startTime = time.time()
        while time.time() - startTime < 0.95:
            self.iteration(mct)
        nextState = mct.tree.chooseChild().value[0]
        (x2, y2) = nextState.getAgentPosition(self.index)
        if x1 + 1 == x2:
            return Directions.EAST
        if x1 - 1 == x2:
            return Directions.WEST
        if y1 + 1 == y2:
            return Directions.NORTH
        if y1 - 1 == y2:
            return Directions.SOUTH
        return Directions.STOP

    # 迭代树一次: selection -> expand -> rollout -> back-propagation
    def iteration(self, mct):
        if mct.tree.children == []:
            self.expand(mct, mct.tree)
        else:
            leaf = mct.select()
            if leaf.value[2] == 0:
                r = self.OfsRollout(leaf.value[0])
                mct.backPropagate(r, leaf)
            elif leaf.value[2] == 1:
                self.expand(mct, leaf)
                newLeaf = random.choice(leaf.children)
                r = self.OfsRollout(newLeaf.value[0])
                mct.backPropagate(r, newLeaf)

    # MCT 的子函数，用于扩展已访问的叶节点
    def expand(self, mct, node):
        actions = node.value[0].getLegalActions(self.index)
        actions.remove(Directions.STOP)
        for action in actions:
            successor = node.value[0].generateSuccessor(self.index, action)
            successorNode = Node((successor, 0, 0))
            mct.insert(node, successorNode)


class OffensiveReflexAgent(ReflexCaptureAgent):

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)  # 后继状态的游戏状态
        curPos = gameState.getAgentState(self.index).getPosition()  # 进攻代理的当前位置
        myPos = successor.getAgentState(
            self.index).getPosition()  # 进攻代理的后继状态位置（如果下一步会死亡，这将是起点）
        nextPosition = nextPos(curPos,
                               action)  # 进攻代理的下一步位置（如果下一步会死亡，这将是下一步位置）
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghost = [a for a in enemies if not a.isPacman and a.getPosition() is not None and manhattanDist(curPos,
                                                                                                        a.getPosition()) <= 5]  # 只计算靠近进攻代理的幽灵
        scaredGhost = [a for a in ghost if a.scaredTimer > 1]  # 被吓到的幽灵但剩余吓唬时间 > 1
        activeGhost = [a for a in ghost if
                       a not in scaredGhost]  # 没有被吓到的幽灵或即将恢复活跃的幽灵（剩余吓唬时间 < 1）
        invaders = [a for a in enemies if
                    a.isPacman and a.getPosition() is not None]  # 在我们的区域发现的入侵者是吃豆人
        currentFoodList = self.getFood(gameState).asList()  # 存储所有当前的点
        openRoadFood = [a for a in currentFoodList if a not in tunnels]  # 存储所有不在隧道中的点
        tunnelFood = [a for a in currentFoodList if a in tunnels]  # 存储所有在隧道中的点
        rev = Directions.REVERSE[
            gameState.getAgentState(self.index).configuration.direction]  # 存储前一个动作的反向动作
        capsule = self.getCapsules(gameState)  # 存储所有当前的胶囊
        checkTunnel = self.ifWasteTunnel(gameState,
                                         successor)  # 每次检查代理是否在隧道入口并评估该隧道

        features['successorScore'] = self.getScore(successor)

        # 如果没有附近的幽灵，设置这些为None，只专注于吃最近的点
        if len(ghost) == 0:
            self.capsule = None
            self.nextOpenFood = None
            self.nextTunnelFood = None

        # 如果代理已经进入对方区域，改为False
        if gameState.getAgentState(self.index).isPacman:
            self.changeEntrance = False

        # 计算代理携带的点数，如果返回则改为0
        if nextPosition in currentFoodList:
            self.carriedDot += 1
        if not gameState.getAgentState(self.index).isPacman:
            self.carriedDot = 0

        # 如果剩余时间仅够返回，添加此特征以强制代理返回
        if self.getTimeLeft(gameState) / 4 < self.getLengthToHome(gameState) + 3:
            features['distToHome'] = self.getLengthToHome(successor)
            return features

        # 当没有附近的幽灵时，吃豆人与最近食物的距离
        if len(activeGhost) == 0 and len(currentFoodList) != 0 and len(currentFoodList) >= 3:
            features['safeFoodDist'] = min([self.getMazeDistance(myPos, food) for food in currentFoodList])
            if myPos in self.getFood(gameState).asList():
                features['safeFoodDist'] = -1

        # 已经可以获胜，让代理安全返回
        if len(currentFoodList) < 3:
            features['return'] = self.getLengthToHome(successor)

            # 吃豆人与最近活跃幽灵的距离
        # 调整为100-距离以避免出现负数
        if len(activeGhost) > 0 and len(currentFoodList) >= 3:
            dists = min([self.getMazeDistance(myPos, a.getPosition()) for a in activeGhost])
            features['distToGhost'] = 100 - dists
            ghostPos = [a.getPosition() for a in activeGhost]
            # 当下一步位置是幽灵位置或幽灵邻居位置时
            if nextPosition in ghostPos:
                features['die'] = 1
            if nextPosition in [getSuccsorsPos(p, legalPositions) for p in ghostPos][0]:
                features['die'] = 1
            # 吃豆人与最近开放道路食物的距离
            if len(openRoadFood) > 0:
                features['openRoadFood'] = min([self.getMazeDistance(myPos, food) for food in openRoadFood])
                if myPos in openRoadFood:
                    features['openRoadFood'] = -1
            elif len(openRoadFood) == 0:
                features['return'] = self.getLengthToHome(successor)

        # 找到所有不在隧道中的安全食物，并获取最近的安全食物
        if len(activeGhost) > 0 and len(currentFoodList) >= 3:
            if len(openRoadFood) > 0:
                safeFood = []
                for food in openRoadFood:
                    if self.getMazeDistance(curPos, food) < min(
                            [self.getMazeDistance(a.getPosition(), food) for a in activeGhost]):
                        safeFood.append(food)
                if len(safeFood) != 0:
                    closestSFdist = min([self.getMazeDistance(curPos, food) for food in safeFood])
                    for food in safeFood:
                        if self.getMazeDistance(curPos, food) == closestSFdist:
                            self.nextOpenFood = food
                            break

        # 找到所有在隧道中的安全食物，并获取最近的安全食物
        if len(activeGhost) > 0 and len(tunnelFood) > 0 and len(scaredGhost) == 0 and len(currentFoodList) >= 3:
            minTFDist = min([self.getMazeDistance(curPos, tf) for tf in tunnelFood])
            safeTfood = []
            for tf in tunnelFood:
                tunnelEntry = getTunnelEntry(tf, tunnels, legalPositions)
                if self.getMazeDistance(curPos, tf) + self.getMazeDistance(tf, tunnelEntry) < min(
                        [self.getMazeDistance(a.getPosition(), tunnelEntry) for a in activeGhost]):
                    safeTfood.append(tf)
            if len(safeTfood) > 0:
                closestTFdist = min([self.getMazeDistance(curPos, food) for food in safeTfood])
                for food in safeTfood:
                    if self.getMazeDistance(curPos, food) == closestTFdist:
                        self.nextTunnelFood = food
                        break

        # 强制Pacman跑向最近的安全食物
        if self.nextOpenFood != None:
            features['goToSafeFood'] = self.getMazeDistance(myPos, self.nextOpenFood)
            if myPos == self.nextOpenFood:
                features['goToSafeFood'] = -0.0001
                self.nextOpenFood = None

        if features['goToSafeFood'] == 0 and self.nextTunnelFood != None:
            features['goToSafeFood'] = self.getMazeDistance(myPos, self.nextTunnelFood)
            if myPos == self.nextTunnelFood:
                features['goToSafeFood'] = 0
                self.nextTunnelFood = None

        # 如果附近有活跃的幽灵并且有胶囊，找到最近的安全胶囊
        if len(activeGhost) > 0 and len(capsule) != 0:
            for c in capsule:
                if self.getMazeDistance(curPos, c) < min(
                        [self.getMazeDistance(c, a.getPosition()) for a in activeGhost]):
                    self.capsule = c

        if len(scaredGhost) > 0 and len(capsule) != 0:
            for c in capsule:
                if self.getMazeDistance(curPos, c) >= scaredGhost[0].scaredTimer and self.getMazeDistance(curPos,
                                                                                                          c) < min(
                        [self.getMazeDistance(c, a.getPosition()) for a in scaredGhost]):
                    self.capsule = c

        if curPos in tunnels:
            for c in capsule:
                if c in getATunnel(curPos, tunnels):
                    self.capsule = c

        # 如果找到最近的安全胶囊，并且被幽灵追赶，跑向那个胶囊
        if self.capsule != None:
            features['distanceToCapsule'] = self.getMazeDistance(myPos, self.capsule)
            if myPos == self.capsule:
                features['distanceToCapsule'] = 0
                self.capsule = None

        # 如果没有附近的活跃幽灵，代理不会吃那个胶囊，除非它挡住了路
        if len(activeGhost) == 0 and myPos in capsule:
            features['leaveCapsule'] = 0.1

        # 通常没有用，当Pacman需要停下来让幽灵走一步以避免死亡时，这个特征会出现
        if action == Directions.STOP: features['stop'] = 1

        # 当在隧道入口时，这个特征强制Pacman不进入一个空的隧道
        if successor.getAgentState(self.index).isPacman and curPos not in tunnels and \
                successor.getAgentState(self.index).getPosition() in tunnels and checkTunnel == 0:
            features['noFoodTunnel'] = -1

        # 当附近有幽灵并且Pacman在隧道入口时，如果这个隧道内有食物，它会计算距离以判断是否可以吃食物然后安全离开隧道。如果不可以，这个特征会出现以强制它离开隧道
        if len(activeGhost) > 0:
            dist = min([self.getMazeDistance(curPos, a.getPosition()) for a in activeGhost])
            if checkTunnel != 0 and checkTunnel * 2 >= dist - 1:
                features['wasteAction'] = -1

        if len(scaredGhost) > 0:
            dist = min([self.getMazeDistance(curPos, a.getPosition()) for a in scaredGhost])
            if checkTunnel != 0 and checkTunnel * 2 >= scaredGhost[0].scaredTimer - 1:
                features['wasteAction'] = -1

        # 当Pacman在隧道中并且突然发现附近的幽灵时，Pacman会判断何时应该离开这个隧道
        if curPos in tunnels and len(activeGhost) > 0:
            foodPos = self.getTunnelFood(gameState)
            if foodPos == None:
                features['escapeTunnel'] = self.getMazeDistance(nextPos(curPos, action), self.tunnelEntry)
            else:
                lengthToEscape = self.getMazeDistance(myPos, foodPos) + self.getMazeDistance(foodPos, self.tunnelEntry)
                ghostToEntry = min([self.getMazeDistance(self.tunnelEntry, a.getPosition()) for a in activeGhost])
                if ghostToEntry - lengthToEscape <= 1 and len(scaredGhost) == 0:
                    features['escapeTunnel'] = self.getMazeDistance(nextPos(curPos, action), self.tunnelEntry)

        if curPos in tunnels and len(scaredGhost) > 0:
            foodPos = self.getTunnelFood(gameState)
            if foodPos == None:
                features['escapeTunnel'] = self.getMazeDistance(nextPos(curPos, action), self.tunnelEntry)
            else:
                lengthToEscape = self.getMazeDistance(myPos, foodPos) + self.getMazeDistance(foodPos, self.tunnelEntry)
                if scaredGhost[0].scaredTimer - lengthToEscape <= 1:
                    features['escapeTunnel'] = self.getMazeDistance(nextPos(curPos, action), self.tunnelEntry)

        if not gameState.getAgentState(self.index).isPacman and len(activeGhost) > 0 and self.stuckStep != -1:
            self.stuckStep += 1

        if gameState.getAgentState(self.index).isPacman or myPos == self.nextEntrance:
            self.stuckStep = 0
            self.nextEntrance = None

        if self.stuckStep > 10:
            self.stuckStep = -1
            self.nextEntrance = random.choice(self.getEntrance(gameState))

            # 当Pacman被幽灵困在边界之间时，10步之后，这个特征会出现以强制Pacman改变另一个随机选择的入口
        if self.nextEntrance != None and features['goToSafeFood'] == 0:
            features['runToNextEntrance'] = self.getMazeDistance(myPos, self.nextEntrance)

        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1, 'distToHome': -100, 'safeFoodDist': -2, 'openRoadFood': -3, 'distToGhost': -10,
                'die': -1000, 'goToSafeFood': -11, 'distanceToCapsule': -1200,
                'return': -1, 'leaveCapsule': -1, 'stop': -50, 'noFoodTunnel': 100, 'wasteAction': 100,
                'escapeTunnel': -1001, 'runToNextEntrance': -1001}
    # 回家的距离
    def getLengthToHome(self, gameState):
        curPos = gameState.getAgentState(self.index).getPosition()
        width = gameState.data.layout.width
        height = gameState.data.layout.height
        legalPositions = [p for p in gameState.getWalls().asList(False)]
        legalRed = [p for p in legalPositions if p[0] == width / 2 - 1]
        legalBlue = [p for p in legalPositions if p[0] == width / 2]
        if self.red:
            return min([self.getMazeDistance(curPos, a) for a in legalRed])
        else:
            return min([self.getMazeDistance(curPos, a) for a in legalBlue])


class DefensiveReflexAgent(ReflexCaptureAgent):

    # getLengthToBoundary: 返回代理到最近边界的距离
    def getLengthToBoundary(self, gameState):
        curPos = gameState.getAgentState(self.index).getPosition()
        width = gameState.data.layout.width
        height = gameState.data.layout.height
        legalPositions = [p for p in gameState.getWalls().asList(False)]
        legalRed = [p for p in legalPositions if p[0] == width / 2 - 1]
        legalBlue = [p for p in legalPositions if p[0] == width / 2]
        if self.red:
            return min([self.getMazeDistance(curPos, a) for a in legalRed])
        else:
            return min([self.getMazeDistance(curPos, a) for a in legalBlue])

    def getFeatures(self, gameState, action):

        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        curPos = gameState.getAgentState(self.index).getPosition()  # 当前防御代理位置
        curState = gameState.getAgentState(self.index)  # 当前防御代理状态
        sucState = successor.getAgentState(self.index)  # 下一个防御代理状态
        sucPos = sucState.getPosition()  # 代理下一个状态的位置
        curCapsule = self.getCapsulesYouAreDefending(gameState)  # 当前胶囊
        lengthToBoundary = self.getLengthToBoundary(successor)  # 到最近边界位置的距离

        # 这个特征强制我们的防御代理只在防御区域内行走
        features['onDefense'] = 100
        if sucState.isPacman: features['onDefense'] = 0

        # 开始时，我们的防御代理会尽快跑到边界
        if self.runToBoundary == None:
            features['runToBoundary'] = self.getLengthToBoundary(successor)

        if self.getLengthToBoundary(successor) <= 2:
            self.runToBoundary = 0

        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]  # 下一个状态的敌人
        curEnemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]  # 当前敌人
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]  # 下一个状态的入侵者
        curInvaders = [a for a in curEnemies if a.isPacman and a.getPosition() != None]  # 当前入侵者

        # 当幽灵可以封锁隧道使Pacman被困时，这个特征强制代理跑到隧道入口
        if self.invadersGuess:
            self.enemyGuess.observe(self, gameState)
            enemyPos = self.enemyGuess.getPossiblePosition(curInvaders[0])
            features['runToTunnelEntry'] = self.getMazeDistance(enemyPos, sucPos)
            self.enemyGuess.elapseTime()

        if self.ifNeedsBlockTunnel(curInvaders, curPos, curCapsule) and curState.scaredTimer == 0:
            features['runToTunnelEntry'] = self.getMazeDistance(
                getTunnelEntry(curInvaders[0].getPosition(), tunnels, legalPositions), sucPos)
            return features

        # 当在隧道中且附近没有入侵者时，会尽快离开隧道
        if curPos in defensiveTunnels and len(curInvaders) == 0:
            features['leaveTunnel'] = self.getMazeDistance(self.start, sucPos)

        # 这个特征会让幽灵尝试杀死Pacman
        features['numInvaders'] = len(invaders)

        # 这个特征强制幽灵在没有发现入侵者时不进入隧道
        if len(curInvaders) == 0 and not successor.getAgentState(self.index).isPacman and curState.scaredTimer == 0:
            if curPos not in defensiveTunnels and successor.getAgentState(self.index).getPosition() in defensiveTunnels:
                features['wasteAction'] = -1

        # features['invaderDistance']: 我的幽灵与最近的入侵者之间的距离
        # features['lengthToBoundary']: 到最近边界位置的距离
        # 这个特征会在幽灵追逐Pacman时出现，确保幽灵在追逐时避免Pacman返回基地
        if len(invaders) > 0 and curState.scaredTimer == 0:
            dists = [self.getMazeDistance(sucPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)
            features['lengthToBoundary'] = self.getLengthToBoundary(successor)

        # 当幽灵处于惊吓状态时，会尝试与Pacman保持两个距离的距离
        if len(invaders) > 0 and curState.scaredTimer != 0:
            dists = min([self.getMazeDistance(sucPos, a.getPosition()) for a in invaders])
            features['followMode'] = (dists - 2) * (dists - 2)
            if curPos not in defensiveTunnels and successor.getAgentState(self.index).getPosition() in defensiveTunnels:
                # 这个特征强制幽灵在没有发现入侵者时不进入隧道
                features['wasteAction'] = -1

        # 当附近有入侵者时，代理与胶囊之间的距离
        if len(invaders) > 0 and len(curCapsule) != 0:
            dist2 = [self.getMazeDistance(c, sucPos) for c in curCapsule]
            features['protectCapsules'] = min(dist2)

        # 当幽灵可以封锁隧道使Pacman被困时，这个特征强制代理停止
        if action == Directions.STOP: features['stop'] = 1

        # 这个特征让我们的幽灵不走回头路
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        # 当幽灵发现我们的食物丢失时，会跑到丢失食物的地方
        if self.getPreviousObservation() != None:
            if len(invaders) == 0 and self.ifLostFood() != None:
                self.curLostFood = self.ifLostFood()

            if self.curLostFood != None and len(invaders) == 0:
                features['goToLostFood'] = self.getMazeDistance(sucPos, self.curLostFood)

            if sucPos == self.curLostFood or len(invaders) > 0:
                self.curLostFood = None

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -100, 'onDefense': 10, 'invaderDistance': -10, 'stop': -100, 'reverse': -2,
                'lengthToBoundary': -3,
                'protectCapsules': -3, 'wasteAction': 200, 'followMode': -100, 'runToTunnelEntry': -10,
                'leaveTunnel': -0.1, 'runToBoundary': -2, 'goToLostFood': -1}

    """
  这个函数用于检查我们的代理是否需要封锁隧道，
  True表示需要封锁。
  """

    def ifNeedsBlockTunnel(self, curInvaders, currentPostion, curCapsule):
        if len(curInvaders) == 1:
            invadersPos = curInvaders[0].getPosition()
            if invadersPos in tunnels:
                tunnelEntry = getTunnelEntry(invadersPos, tunnels, legalPositions)
                if self.getMazeDistance(tunnelEntry, currentPostion) <= self.getMazeDistance(tunnelEntry,
                                                                                             invadersPos) and curCapsule not in getATunnel(
                        invadersPos, tunnels):
                    return True
        return False

    """
  这个函数用于检查我们当前是否有食物丢失
  """

    def ifLostFood(self):
        preState = self.getPreviousObservation()
        currState = self.getCurrentObservation()
        myCurrFood = self.getFoodYouAreDefending(currState).asList()
        myLastFood = self.getFoodYouAreDefending(preState).asList()
        if len(myCurrFood) < len(myLastFood):
            for i in myLastFood:
                if i not in myCurrFood:
                    return i
        return None


