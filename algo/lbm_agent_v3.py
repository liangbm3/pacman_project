
from captureAgents import CaptureAgent
import distanceCalculator
import random
import time
import util
import sys
from game import Directions
import game
from util import nearestPoint
from game import Grid
import math

#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed,
               first='ReflexCaptureAgent', second='ReflexCaptureAgent'):
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
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


def getAggression(gameState, pos, isRed):
    """
    根据位置和游戏状态计算攻击等级。

    Parameters:
    - gameState: 当前游戏状态。
    - pos: 代理的位置。
    - isRed: 一个标志，指示agent是否在红队。

    Returns:
    - float: agent的攻击等级。

    """
    walls = gameState.getWalls()
    x, y = pos
    if isRed:
        return 1.0 * x / walls.width
    else:
        return 1.0 * (walls.width - x) / walls.width

def getDeepFood(gameState, agent, isRed):
    """
    根据 isRed 标志，返回食物网格的浅表副本，其中某些单元设置为 False。

    Args:
        gameState (GameState): 游戏状态·
        agent (Agent): 自定义agent类
        isRed (bool): 指示agent是否属于红队的标志。

    Returns:
        shallowFood (Grid): 代表食物网格的浅表副本。
    """
    food = agent.getFood(gameState)
    shallowFood = food.copy()
    if isRed:
        shallowDistThresholdCol = int(0.8 * food.width)
        for c in range(shallowDistThresholdCol):
            for r in range(food.height):
                shallowFood[c][r] = False
    else:
        shallowDistThresholdCol = int(0.2 * food.width)
        for c in range(shallowDistThresholdCol, food.width):
            for r in range(food.height):
                shallowFood[c][r] = False
    return shallowFood

def getShallowFood(gameState, agent, isRed):
    """
    返回食物网格的浅表副本，其中根据 isRed 标志，只有网格的一部分可见。

    Args:
        gameState (object): 当前游戏状态。
        agent (object): 我们自定义的agent类。
        isRed (bool): 代表团队是否为红队。

    Returns:
        list: 代表食物网格的浅表副本。
    """
    food = agent.getFood(gameState)
    deepFood = food.copy()
    if isRed:
        deepDistThresholdCol = int(0.8 * food.width)
        for c in range(deepDistThresholdCol, food.width):
            for r in range(food.height):
                deepFood[c][r] = False
    else:
        deepDistThresholdCol = int(0.2 * food.width)
        for c in range(deepDistThresholdCol):
            for r in range(food.height):
                deepFood[c][r] = False
    return deepFood

def getMinDistToCapsule(gameState, agent, pos, capsules):
    """
    计算从给定位置到最近的胶囊的最小距离。

    Args:
      gameState (object): 当前游戏状态。
      agent (object): 我们定义的agent对象。
      pos (tuple): 代表位置的元组。
      capsules (list): 代表胶囊位置的元组列表。

    Returns:
      int: 从给定位置到最近的胶囊的最小距离。
    """
    # 如果没有胶囊，则返回0
    if len(capsules) == 0:
        return 0
    minDist = 1000
    for capsule in capsules:
        dist = agent.getMazeDistance(pos, capsule)
        if dist < minDist:
            minDist = dist
    return minDist

def getCloseSafePoints(gameState, isRed):
    """
    返回靠近游戏板中心列的安全点列表。
    Parameters:
    - gameState: 当前游戏状态。
    - isRed: 一个标志，指示agent是否在红队。

    Returns:
    - safePoints: 
    """
    walls = gameState.getWalls()
    if isRed:
        col = int((walls.width / 2)) - 1
    else:
        col = int((walls.width / 2))
    safePoints = []
    for y in range(walls.height):
        if not walls[col][y]:
            safePoints.append((col, y))
    return safePoints

def getPositionAfterAction(pos, action):
    """
      接收当前位置和动作，并返回新位置。

    Parameters:
    - pos (tuple): 代表位置的元组。
    - action (str): 代表动作的字符串。

    Returns:
    - tuple: 代表新位置的元组。
    """
    x, y = pos
    if action == 'East':
        return (x + 1, y)
    if action == 'West':
        return (x - 1, y)
    if action == 'North':
        return (x, y + 1)
    if action == 'South':
        return (x, y - 1)
    else:
        return pos

def getPossibleActions(gameState, pos):
    """
      接收游戏状态和位置，并确定agent可能的动作。

    Args:
      gameState (object): 目前的游戏状态。
      pos (tuple): 代表位置的元组。

    Returns:
      list: 代表可能动作的字符串列表。

    """
    x, y = pos
    walls = gameState.getWalls()
    actions = list()
    if not walls[x + 1][y]:
        actions.append('East')
    if not walls[x - 1][y]:
        actions.append('West')
    if not walls[x][y + 1]:
        actions.append('North')
    if not walls[x][y - 1]:
        actions.append('South')
    actions.append('Stop')
    return actions

def getDeadEnds(gameState):
    """
      接收游戏状态，并在游戏状态中找出死胡同。

    Parameters:
    - gameState: 当前游戏状态。

    Returns:
    - deadEnds: 代表游戏中死胡同的列表。
    """
    walls = gameState.getWalls()
    deadEnds = []
    for r in range(walls.height):
        for c in range(walls.width):
            pos = (c, r)
            if not walls[c][r]:
                if len(getPossibleActions(gameState, pos)) <= 2:  # Stop and one other action
                    deadEnds.append(pos)
    return deadEnds

def getChokePointAndDirection(gameState, deadEnd):
    """
    接收游戏状态和死胡同的位置，并找出阻塞点和方向以到达它。

    Args:
      gameState (object): 当前游戏状态。
      deadEnd (tuple): 代表死胡同位置的元组。

    Returns:
      tuple: 代表阻塞点和方向的元组。
    """
    walls = gameState.getWalls()
    pos = deadEnd
    actions = getPossibleActions(gameState, pos)  # 获得在死胡同中能作出动作的列表
    action = actions[0]  # 获得第一个动作，死胡同只能作出一个动作，因此该动作就是唯一的动作
    newPos = getPositionAfterAction(pos, action)  # 执行动作后的位置
    actions = getPossibleActions(gameState, newPos)  # 获得在新位置能作出动作的列表
    actions.remove('Stop')  # 去掉停止动作

    # 进入一个循环，找到一个可以作出多于一个选择的位置，除了前进和后退之外，还有其他选择
    while len(actions) == 2:
        newestPos1 = getPositionAfterAction(newPos, actions[0])
        newestPos2 = getPositionAfterAction(newPos, actions[1])
        if newestPos1 == pos:
            newestPos = newestPos2
        else:
            newestPos = newestPos1
        pos = newPos
        newPos = newestPos
        actions = getPossibleActions(gameState, newPos)
        actions.remove('Stop')

    # 新的阻塞点的前一个位置
    dirPos = pos
    # 新的阻塞点
    chokePoint = newPos
    # 寻找到达阻塞点的方向
    actions = getPossibleActions(gameState, newPos)
    for action in actions:
        if getPositionAfterAction(chokePoint, action) == dirPos:
            dirAction = action
            break
    return chokePoint, dirAction

def getChokePointAndDirectionRestricted(gameState, deadEnd, directionsRestricted):
    """
        根据给定的游戏状态、死胡同位置和限制方向，找到阻塞点和限制方向。

        Parameters:
        - gameState (object): 当前游戏状态。
        - deadEnd (tuple): 代表死胡同位置的元组。
        - directionsRestricted (list): 代表限制方向的字符串列表。

        Returns:
        - chokePoint (tuple): 代表阻塞点的元组。
        - dirAction (str): 代表到达阻塞点的方向。

    """
    walls = gameState.getWalls()
    pos = deadEnd
    actions = getPossibleActions(gameState, pos)
    for action in actions:
        if action in directionsRestricted or action == 'Stop':
            actions.remove(action)
    action = actions[0]
    newPos = getPositionAfterAction(pos, action)
    actions = getPossibleActions(gameState, newPos)
    actions.remove('Stop')
    while len(actions) == 2:
        newestPos1 = getPositionAfterAction(newPos, actions[0])
        newestPos2 = getPositionAfterAction(newPos, actions[1])
        if newestPos1 == pos:
            newestPos = newestPos2
        else:
            newestPos = newestPos1
        pos = newPos
        newPos = newestPos
        actions = getPossibleActions(gameState, newPos)
        actions.remove('Stop')
    dirPos = pos
    chokePoint = newPos
    actions = getPossibleActions(gameState, newPos)
    for action in actions:
        if getPositionAfterAction(chokePoint, action) == dirPos:
            dirAction = action
            break
    return chokePoint, dirAction

def getChokePoints(gameState):
    """
    在游戏状态中找出阻塞点。

    Parameters:
        - gameState: 当前游戏状态。

    Returns:
        - chokePointList: 代表游戏中阻塞点的列表。

    """
    walls = gameState.getWalls()
    deadEnds = getDeadEnds(gameState)
    chokePointList = []

    # 记录阻塞点的字典，键为阻塞点，值为到达阻塞点的方向
    chokePoints = {}
    # 记录待检查的阻塞点的字典，键为阻塞点，值为到达阻塞点的方向
    toCheck = {}
    stillGoing = False

    # 遍历每一个死胡同
    for deadEnd in deadEnds:
        chokePoint, direction = getChokePointAndDirection(gameState, deadEnd)
        if not chokePoint in chokePoints:
            chokePoints[chokePoint] = []
            chokePoints[chokePoint].append(direction)
        # 如果从一个窒息点出发的可能动作数减去2（考虑到原地不动的选项）等于该点方向列表的长度，
        # 这意味着我们已经找到了所有从该窒息点出发的方向，
        # 这个点就被认为是一个待检查的窒息点，并从当前的窒息点字典中移除，添加到toCheck字典中。
        if len(getPossibleActions(gameState, chokePoint)) - 2 == len(chokePoints[chokePoint]):
            toCheck[chokePoint] = chokePoints[chokePoint]
            del chokePoints[chokePoint]
            stillGoing = True
    for chokePoint in chokePoints:
        chokePointList.append((chokePoint, chokePoints[chokePoint]))

    # 如果还有窒息点需要检查，继续检查
    while stillGoing:
        deadEnds = toCheck.copy()
        toCheck = {}
        chokePoints = {}
        stillGoing = False
        for deadEnd in deadEnds:
            chokePoint, direction = getChokePointAndDirectionRestricted(
                gameState, deadEnd, deadEnds[deadEnd])
            if not chokePoint in chokePoints:
                chokePoints[chokePoint] = []
            chokePoints[chokePoint].append(direction)
            if len(getPossibleActions(gameState, chokePoint)) - 2 == len(chokePoints[chokePoint]):
                toCheck[chokePoint] = chokePoints[chokePoint]
                chokePoints = chokePoints.pop(chokePoint)
                stillGoing = True
        for chokePoint in chokePoints:
            chokePointList.append((chokePoint, chokePoints[chokePoint]))

    return chokePointList

def getMinDistChokePoints(gameState, agent, pos, chokePoints):
    """
    计算位置到阻塞点的最小距离。

    Parameters:
    - gameState: 当前游戏状态。
    - agent: 代理对象。
    - pos: 位置。
    - chokePoints: 阻塞点。

    Returns:
    - minDist: 最小距离。
    """
    minDist = 1000
    walls = gameState.getWalls()
    for choke, dirs in chokePoints:
        dist = agent.getMazeDistance(pos, choke)
        if dist < minDist:
            minDist = dist
    return minDist

def bfsDepthGrid(gameState, chokePoint, direction, depthGrid):
    """
    执行广度优先搜索，以计算给定阻塞点的深度网格。

    Args:
        gameState (object): 当前游戏状态
        chokePoint (tuple): 开始搜索的阻塞点。
        direction (str): 从阻塞点移动的方向。
        depthGrid (list): 代表游戏中每个单元深度的网格。

    Returns:
        list: 代表游戏中每个单元深度的网格。
    """
    x, y = chokePoint
    depthGrid[x][y] = 9
    newPos = getPositionAfterAction(chokePoint, direction)
    posQueue = util.Queue()
    posQueue.push((newPos, 1))
    while not posQueue.isEmpty():
        pos, depth = posQueue.pop()
        x, y = pos
        if depthGrid[x][y] == 0:
            depthGrid[x][y] = depth
            actions = getPossibleActions(gameState, pos)
            for action in actions:
                nextPos = getPositionAfterAction(pos, action)
                newDepth = depth + 1
                posQueue.push((nextPos, newDepth))
    x, y = chokePoint
    depthGrid[x][y] = 0
    return depthGrid

def getDepthsAndChokePoints(gameState):
    """
    接收游戏状态，并在游戏状态中找出阻塞点和深度网格。

    Args:
      gameState: 当前游戏状态。

    Returns:
      depthGrid: 代表游戏中每个单元深度的网格。
      chokePoints: 代表游戏中窒息点的列表。
    """
    walls = gameState.getWalls() # 获取墙壁信息
    depthGrid = Grid(walls.width, walls.height, initialValue=0)# 创建一个网格，用于存储每个单元的深度

    chokePoints = getChokePoints(gameState)
    for chokePoint, directions in chokePoints:
        for direction in directions:
            depthGrid = bfsDepthGrid(
                gameState, chokePoint, direction, depthGrid)

    return depthGrid, chokePoints

def getDistToSafety(gameState, agent):
    """
      计算到最近安全点的距离。
    Args:
      gameState: 当前游戏状态。
      agent: 我们定义的agent对象。

    Returns:
      int: 到最近安全点的距离。
    """
    agentIndex = agent.index
    isPacman = gameState.getAgentState(agentIndex).isPacman
    # if not isPacman:
    #  return 0
    isRed = agent.red
    pos = gameState.getAgentPosition(agentIndex)
    closestSafePoints = getCloseSafePoints(gameState, isRed)
    closestDist = 1000
    for point in closestSafePoints:
        dist = agent.getMazeDistance(pos, point)
        if dist < closestDist:
            closestDist = dist
    return closestDist

def getDistAdvantage(gameState, agent, enemyPos, isDefending):
    """
    计算agent的距离优势。
    Parameters:
    - gameState: 当前游戏状态。
    - agent:    我们定义的agent对象。
    - enemyPos: 敌人的位置。
    - isDefending: 一个标志，指示agent是否在防守。
    Returns:
    - int: 距离优势。
    """
    agentIndex = agent.index
    isRed = agent.red
    pos = gameState.getAgentPosition(agentIndex)

    if isDefending:
        closestSafePoints = getCloseSafePoints(gameState, not isRed)
        minAdvantage = 1000
    else:
        closestSafePoints = getCloseSafePoints(gameState, isRed)
        maxAdvantage = -1000
    if len(closestSafePoints) == 0:
        return 0
    for point in closestSafePoints:
        dist = agent.getMazeDistance(pos, point)
        enemyDist = agent.getMazeDistance(enemyPos, point)
        advantage = enemyDist - dist
        if isDefending:
            if advantage < minAdvantage:
                minAdvantage = advantage
        else:
            if advantage > maxAdvantage:
                maxAdvantage = advantage
    if isDefending:
        return minAdvantage
    else:
        return maxAdvantage

def getDistAdvantageCapsule(gameState, agent, enemyPos, isDefending):
    """
    返回胶囊的距离优势。

    Args:
        gameState (object): 当前游戏状态。
        agent (object): 我们定义的代理对象。
        enemyPos (tuple): 敌人的位置。
        isDefending (bool): 一个标志，指示代理是否在防守。

    Returns:
        int: 胶囊的距离优势。

  """
    agentIndex = agent.index
    isRed = agent.red
    pos = gameState.getAgentPosition(agentIndex)
    minAdvantage = 1000
    # 如果在防守，则考虑防御的胶囊，否则考虑游戏中的所有胶囊
    if isDefending:
        closestSafePoints = agent.getCapsulesYouAreDefending(gameState)
        minAdvantage = 1000
    else:
        closestSafePoints = agent.getCapsules(gameState)
        maxAdvantage = -1000
    # 如果没有安全的胶囊，则返回0
    if len(closestSafePoints) == 0:
        return 0

    # 优势的计算方法，敌人到胶囊的距离减去agent到胶囊的距离
    for point in closestSafePoints:
        dist = agent.getMazeDistance(pos, point)
        enemyDist = agent.getMazeDistance(enemyPos, point)
        advantage = enemyDist - dist
        # 如果在防守，则找到最小的优势，否则找到最大的优势
        if isDefending:
            if advantage < minAdvantage:
                minAdvantage = advantage
        else:
            if advantage > maxAdvantage:
                maxAdvantage = advantage
    if isDefending:
        return minAdvantage
    else:
        return maxAdvantage

enemyLocationPredictor = None
PERSIST = 10

class EnemyLocationPredictor:


    def __init__(self, gameState, agent):
        """
        预测吃豆人游戏中敌人位置的类。

        Attributes:
        - depthGrid: 代表游戏中每个单元深度的网格。
        - enemyIndices: 代表敌人代理的索引的列表。
        - teamIndices: 代表团队代理的索引的列表。
        - walls: 代表游戏中墙壁的网格。
        - pastFood: 代表团队正在保卫的食物的网格。
        - pastCapsules: 代表团队正在保卫的胶囊的网格。
        - possiblePositions1: 代表敌人1的可能位置的网格。
        - possiblePositions2: 代表敌人2的可能位置的网格。
        - positionsToInvestigate: 一个位置列表，用于敌人位置的调查。
        - positionsToAvoid: 一个位置列表，用于避免敌人位置。
        - enemies: 代表敌人的索引的列表。
        - isRed: 代表团队是否为红队。
        - pastTeamLocation1: 团队成员1的上一个位置。
        - pastTeamLocation2: 团队成员2的上一个位置。
        - enemyStartPos: 敌人的初始位置。
        - enemy1KnownLocation: 敌人1的已知位置。
        - enemy2KnownLocation: 敌人2的已知位置。
        - ignoreCounterInvestigate: 一个计数器，用于忽略调查位置。
        - ignoreCounterAvoid: 一个计数器，用于忽略避免位置。
        """

        self.depthGrid = agent.depthGrid
        self.enemyIndices = agent.getOpponents(gameState)
        self.teamIndices = agent.getTeam(gameState)
        self.walls = gameState.getWalls()
        self.pastFood = agent.getFoodYouAreDefending(gameState)
        self.pastCapsules = agent.getCapsulesYouAreDefending(gameState)
        self.possiblePositions1 = Grid(self.walls.width, self.walls.height)
        self.possiblePositions2 = Grid(self.walls.width, self.walls.height)
        self.positionsToInvestigate = []
        self.positionsToAvoid = []
        self.enemies = agent.getOpponents(gameState)
        self.isRed = not agent.red
        self.pastTeamLocation1 = agent.start
        self.pastTeamLocation2 = agent.start
        x, y = agent.start
        enemyX = self.walls.width - x - 1
        enemyY = self.walls.height - y - 1
        self.enemyStartPos = (enemyX, enemyY)

        self.enemy1KnownLocation = self.enemyStartPos
        self.enemy2KnownLocation = self.enemyStartPos
        self.possiblePositions1[enemyX][enemyY] = True
        self.possiblePositions2[enemyX][enemyY] = True

        self.ignoreCounterInvestigate = 0
        self.ignoreCounterAvoid = 0

    def getGridWithPositions(self, positions):
        g = Grid(self.walls.width, self.walls.height)
        for pos in positions:
            x, y = pos
            x, y = int(x), int(y)
            g[x][y] = True
        return g

    def adjustConsideration(self, invaders, defenders):
        if len(self.positionsToAvoid) > defenders:
            while len(self.positionsToAvoid) > defenders and len(self.positionsToAvoid) > 0:
                self.positionsToAvoid.pop(0)
        if len(self.positionsToInvestigate) > invaders:
            while len(self.positionsToInvestigate) > invaders and len(self.positionsToInvestigate) > 0:
                self.positionsToInvestigate.pop(0)

        if self.ignoreCounterInvestigate > 0:
            self.ignoreCounterInvestigate -= 1
        elif len(self.positionsToInvestigate) > 0:
            self.ignoreCounterInvestigate = PERSIST
            pos = self.positionsToInvestigate.pop(0)

        if self.ignoreCounterAvoid > 0:
            self.ignoreCounterAvoid -= 1
        elif len(self.positionsToAvoid) > 0:
            self.ignoreCounterAvoid = PERSIST
            pos = self.positionsToAvoid.pop(0)

    def addPositionForConsideration(self, pos, invaders, defenders, isRed):
        x, y = pos

        if (x < (self.walls.width / 2) and isRed) or (x >= (self.walls.width / 2) and not isRed):
            # Add position to avoid

            self.ignoreCounterAvoid = PERSIST

            if not pos in self.positionsToAvoid:
                if len(self.positionsToAvoid) > defenders:
                    while len(self.positionsToAvoid) >= defenders and len(self.positionsToAvoid) > 0:
                        self.positionsToAvoid.pop(0)
                self.positionsToAvoid.append(pos)

        else:
            # Add position to investigate

            self.ignoreCounterInvestigate = PERSIST

            if not pos in self.positionsToInvestigate:
                if len(self.positionsToInvestigate) > invaders:
                    while len(self.positionsToInvestigate) > invaders and len(self.positionsToInvestigate) > 0:
                        self.positionsToInvestigate.pop(0)
                self.positionsToInvestigate.append(pos)

    def expandProb(self, prob):
        newProb = prob.copy()
        for r in range(prob.height):
            for c in range(prob.width):
                if prob[c][r]:
                    newProb[c][r] = True
                    if c > 0:
                        newProb[c - 1][r] = True
                    if c < prob.width - 1:
                        newProb[c + 1][r] = True
                    if r > 0:
                        newProb[c][r - 1] = True
                    if r < prob.height - 1:
                        newProb[c][r + 1] = True
        return newProb

    def updatePart(self, gameState, agent, teamPosition1, teamPosition2, verbose=False):
        invaders = len(
            [a for a in self.enemies if gameState.getAgentState(a).isPacman])
        defenders = len(
            [a for a in self.enemies if not gameState.getAgentState(a).isPacman])
        knownPos1 = None
        knownPos2 = None
        isRed = self.isRed

        possiblePositions1 = self.expandProb(self.possiblePositions1)
        possiblePositions2 = self.expandProb(self.possiblePositions2)

        for r in range(self.walls.height):
            for c in range(self.walls.width):
                if self.walls[c][r]:
                    possiblePositions1[c][r] = False
                    possiblePositions2[c][r] = False

        newFood = agent.getFoodYouAreDefending(gameState)
        newCapsules = agent.getCapsulesYouAreDefending(gameState)
        enemy1State = gameState.getAgentState(self.enemyIndices[0])
        enemy2State = gameState.getAgentState(self.enemyIndices[1])
        enemiesRed = not agent.red
        team1X, team1Y = teamPosition1
        team2X, team2Y = teamPosition2
        eatenFood = []
        eatenCapsules = []

        # If the enemy's position is known, use that. Otherwise, no enemy can be near a team member
        if not gameState.getAgentState(self.enemyIndices[0]).getPosition() is None:
            knownPos1 = gameState.getAgentState(
                self.enemyIndices[0]).getPosition()
        else:
            for r in range(self.walls.height):
                for c in range(self.walls.width):
                    manhattanDist1 = abs(c - team1X) + abs(r - team1Y)
                    manhattanDist2 = abs(c - team2X) + abs(r - team2Y)
                    if min(manhattanDist1, manhattanDist2) <= 5:
                        possiblePositions1[c][r] = False

        if not gameState.getAgentState(self.enemyIndices[1]).getPosition() is None:
            knownPos2 = gameState.getAgentState(
                self.enemyIndices[1]).getPosition()
        else:
            for r in range(self.walls.height):
                for c in range(self.walls.width):
                    manhattanDist1 = abs(c - team1X) + abs(r - team1Y)
                    manhattanDist2 = abs(c - team2X) + abs(r - team2Y)
                    if min(manhattanDist1, manhattanDist2) <= 5:
                        possiblePositions2[c][r] = False

        # If an enemy was captured, it must be at its spawn (POS KNOWN)
        if agent.getMazeDistance(self.pastTeamLocation1, teamPosition1) > 1:
            team1Captured = True
        else:
            team1Captured = False

        if agent.getMazeDistance(self.pastTeamLocation2, teamPosition2) > 1:
            team2Captured = True
        else:
            team2Captured = False

        if not self.enemy1KnownLocation is None and knownPos1 is None:
            dist = min(agent.getMazeDistance(self.enemy1KnownLocation, self.pastTeamLocation1),
                       agent.getMazeDistance(self.enemy1KnownLocation, self.pastTeamLocation2))
            if dist < 3:
                enemy1Captured = True
            else:
                enemy1Captured = False
        else:
            enemy1Captured = False

        if not self.enemy2KnownLocation is None and knownPos2 is None:
            dist = min(agent.getMazeDistance(self.enemy2KnownLocation, self.pastTeamLocation1),
                       agent.getMazeDistance(self.enemy2KnownLocation, self.pastTeamLocation2))
            if dist < 3:
                enemy2Captured = True
            else:
                enemy2Captured = False
        else:
            enemy2Captured = False

        if enemy1Captured:
            self.addPositionForConsideration(
                self.enemyStartPos, invaders, defenders, isRed)
            knownPos1 = self.enemyStartPos
        if enemy2Captured:
            self.addPositionForConsideration(
                self.enemyStartPos, invaders, defenders, isRed)
            knownPos2 = self.enemyStartPos

        # If a team member was captured, assume that the nearest enemy to that team member is now where the team member was. (POS NOT KNOWN)
        # An enemy can't capture a team member without the team member knowing where the enemy is at the time of the capture
        if team1Captured and team2Captured:
            if not self.enemy1KnownLocation is None and not self.enemy2KnownLocation is None:

                dist11 = agent.getMazeDistance(
                    self.enemy1KnownLocation, self.pastTeamLocation1)
                dist12 = agent.getMazeDistance(
                    self.enemy1KnownLocation, self.pastTeamLocation2)
                dist21 = agent.getMazeDistance(
                    self.enemy2KnownLocation, self.pastTeamLocation1)
                dist22 = agent.getMazeDistance(
                    self.enemy2KnownLocation, self.pastTeamLocation2)

                canBothCapture1 = False
                canBothCapture2 = False

                if dist11 < 3 and dist21 < 3:
                    # Both agents could capture team1
                    canBothCapture1 = True
                elif dist11 < 3 and knownPos1 is None:
                    # Enemy1 must have captured team1
                    knownPos1 = self.pastTeamLocation1
                    self.addPositionForConsideration(
                        self.pastTeamLocation1, invaders, defenders, isRed)
                elif dist21 < 3 and knownPos2 is None:
                    # Enemy2 must have captured team1
                    knownPos2 = self.pastTeamLocation1
                    self.addPositionForConsideration(
                        self.pastTeamLocation1, invaders, defenders, isRed)

                if dist12 < 3 and dist22 < 3:
                    # Both agents could capture team2
                    canBothCapture2 = True
                elif dist12 < 3 and knownPos1 is None:
                    # Enemy1 must have captured team2
                    knownPos1 = self.pastTeamLocation2
                    self.addPositionForConsideration(
                        self.pastTeamLocation2, invaders, defenders, isRed)
                elif dist22 < 3 and knownPos2 is None:
                    # Enemy2 must have captured team2
                    knownPos2 = self.pastTeamLocation2
                    self.addPositionForConsideration(
                        self.pastTeamLocation2, invaders, defenders, isRed)

                if canBothCapture1 and canBothCapture2:
                    # Make arbitrary assumption
                    self.addPositionForConsideration(
                        self.pastTeamLocation1, invaders, defenders, isRed)
                    self.addPositionForConsideration(
                        self.pastTeamLocation2, invaders, defenders, isRed)
                elif canBothCapture1:
                    self.addPositionForConsideration(
                        self.pastTeamLocation1, invaders, defenders, isRed)
                elif canBothCapture2:
                    self.addPositionForConsideration(
                        self.pastTeamLocation2, invaders, defenders, isRed)

            elif not self.enemy1KnownLocation is None and knownPos1 is None:
                # Enemy1 Captured both agents at once
                knownPos1 = self.pastTeamLocation1
                self.addPositionForConsideration(
                    self.pastTeamLocation1, invaders, defenders, isRed)
            elif knownPos2 is None:
                # Enemy2 Captured both agents at once
                knownPos2 = self.pastTeamLocation1
                self.addPositionForConsideration(
                    self.pastTeamLocation1, invaders, defenders, isRed)

        elif team1Captured:
            if not self.enemy1KnownLocation is None and not self.enemy2KnownLocation is None:
                dist1 = agent.getMazeDistance(
                    self.enemy1KnownLocation, self.pastTeamLocation1)
                dist2 = agent.getMazeDistance(
                    self.enemy2KnownLocation, self.pastTeamLocation1)

                if dist1 < 3 and dist2 < 3:
                    pass

                elif dist1 < 3 and knownPos1 is None:
                    # Enemy1 must have captured team1
                    knownPos1 = self.pastTeamLocation1
                    self.addPositionForConsideration(
                        self.pastTeamLocation1, invaders, defenders, isRed)

                elif knownPos2 is None:
                    knownPos2 = self.pastTeamLocation1
                    # Enemy2 must have captured team1
                    self.addPositionForConsideration(
                        self.pastTeamLocation1, invaders, defenders, isRed)

            elif not self.enemy1KnownLocation is None and knownPos1 is None:
                # Enemy1 must have captured team1
                knownPos1 = self.pastTeamLocation1
                self.addPositionForConsideration(
                    self.pastTeamLocation1, invaders, defenders, isRed)

            elif knownPos2 is None:
                knownPos2 = self.pastTeamLocation1
                # Enemy2 must have captured team1
                self.addPositionForConsideration(
                    self.pastTeamLocation1, invaders, defenders, isRed)

        elif team2Captured:
            if not self.enemy1KnownLocation is None and not self.enemy2KnownLocation is None:
                dist1 = agent.getMazeDistance(
                    self.enemy1KnownLocation, self.pastTeamLocation2)
                dist2 = agent.getMazeDistance(
                    self.enemy2KnownLocation, self.pastTeamLocation2)

                if dist1 < 3 and dist2 < 3 and knownPos1 is None and knownPos2 is None:
                    # Both agents could capture team1, arbitrary decision
                    self.addPositionForConsideration(
                        self.pastTeamLocation2, invaders, defenders, isRed)

                elif dist1 < 3 and knownPos1 is None:
                    # Enemy1 must have captured team1
                    knownPos1 = self.pastTeamLocation2
                    self.addPositionForConsideration(
                        self.pastTeamLocation2, invaders, defenders, isRed)

                elif dist2 < 3 and knownPos2 is None:
                    # Enemy2 must have captured team1
                    knownPos2 = self.pastTeamLocation2
                    self.addPositionForConsideration(
                        self.pastTeamLocation2, invaders, defenders, isRed)

            elif not self.enemy1KnownLocation is None and knownPos1 is None:
                # Enemy1 must have captured team1
                knownPos1 = self.pastTeamLocation2
                self.addPositionForConsideration(
                    self.pastTeamLocation2, invaders, defenders, isRed)

            elif knownPos2 is None:
                # Enemy2 must have captured team1
                knownPos2 = self.pastTeamLocation2
                self.addPositionForConsideration(
                    self.pastTeamLocation2, invaders, defenders, isRed)

        # If food was eaten, assume that the nearest enemy to that food is now where the food was. Otherwise, no enemy can be at a food location (POS NOT KNOWN)

        if len(self.pastFood.asList()) > len(newFood.asList()):
            for r in range(self.walls.height):
                for c in range(self.walls.width):
                    if self.pastFood[c][r] and not newFood[c][r]:
                        pos = (c, r)
                        eatenFood.append(pos)
            # for each food eaten, get the total probability of spaces surrounding the food for both enemies.
            for pos in eatenFood:
                self.addPositionForConsideration(
                    pos, invaders, defenders, isRed)

        for r in range(self.walls.height):
            for c in range(self.walls.width):
                if newFood[c][r]:
                    possiblePositions1[c][r] = False
                    possiblePositions2[c][r] = False

        # If a capsule was eaten, assume that the nearest enemy to that capsule is now where the capsule was. Otherwise, no enemy can be at a capsule location (POS NOT KNOWN)

        if len(self.pastCapsules) > len(newCapsules):
            for capsule in self.pastCapsules:
                if not capsule in newCapsules:
                    eatenCapsules.append(capsule)
            for pos in eatenCapsules:
                self.addPositionForConsideration(
                    pos, invaders, defenders, isRed)

        for pos in newCapsules:
            x, y = pos
            possiblePositions1[x][y] = False
            possiblePositions2[x][y] = False

        # If an enemy is defending, it must be on its side of the field. Otherwise, it must be on your side of the field. (POS NOT KNOWN)
        if enemy1State.isPacman:
            for r in range(self.walls.height):
                for c in range(self.walls.width):
                    if enemiesRed:
                        if c < self.walls.width / 2:
                            possiblePositions1[c][r] = False
                    else:
                        if c >= self.walls.width / 2:
                            possiblePositions1[c][r] = False
        else:
            for r in range(self.walls.height):
                for c in range(self.walls.width):
                    if enemiesRed:
                        if c >= self.walls.width / 2:
                            possiblePositions1[c][r] = False
                    else:
                        if c < self.walls.width / 2:
                            possiblePositions1[c][r] = False

        if enemy2State.isPacman:
            for r in range(self.walls.height):
                for c in range(self.walls.width):
                    if enemiesRed:
                        if c < self.walls.width / 2:
                            possiblePositions2[c][r] = False
                    else:
                        if c >= self.walls.width / 2:
                            possiblePositions2[c][r] = False
        else:
            for r in range(self.walls.height):
                for c in range(self.walls.width):
                    if enemiesRed:
                        if c >= self.walls.width / 2:
                            possiblePositions2[c][r] = False
                    else:
                        if c < self.walls.width / 2:
                            possiblePositions2[c][r] = False

        if not knownPos1 is None:
            possiblePositions1 = self.getGridWithPositions([knownPos1])
        if not knownPos2 is None:
            possiblePositions2 = self.getGridWithPositions([knownPos2])

        self.pastTeamLocation1 = teamPosition1
        self.pastTeamLocation2 = teamPosition2
        self.pastFood = agent.getFoodYouAreDefending(gameState)
        self.pastCapsules = agent.getCapsulesYouAreDefending(gameState)
        self.enemy1KnownLocation = knownPos1
        self.enemy2KnownLocation = knownPos2

        self.adjustConsideration(invaders, defenders)

        return possiblePositions1, possiblePositions2

    def update(self, gameState, agent, teamPosition1, teamPosition2, verbose=False):
        self.possiblePositions1, self.possiblePositions2 = self.updatePart(
            gameState, agent, teamPosition1, teamPosition2, verbose)

    def getPositionPossibleGrid(self):
        g = Grid(self.walls.width, self.walls.height)
        for r in range(self.walls.height):
            for c in range(self.walls.width):
                g[c][r] = self.possiblePositions1[c][r] or self.possiblePositions2[c][r]
        return g

    def getPositionsToInvestigate(self):
        return self.positionsToInvestigate

    def getPositionsToAvoid(self):
        return self.positionsToAvoid

    def removePositionFromInvestigation(self, pos):
        self.positionsToInvestigate.remove(pos)

    def removePositionFromAvoidance(self, pos):
        self.positionsToAvoid.remove(pos)


ENOUGH_DOTS = 4

target1 = None
target2 = None
mode1 = None
mode2 = None
needHelp = False
chargeInvaderCounter = 0
invaderPresentCounter = 0
ignoreCounter1 = 0
ignoreCounter2 = 0
random1 = 0


class ReflexCaptureAgent(CaptureAgent):
    """
    选择得分最大化动作的agent类
    """

    def registerInitialState(self, gameState):
        """
        Initializes the agent's state at the start of the game.

        Args:
          gameState: The current game state.

        Returns:
          None
        """
        global enemyLocationPredictor
        global mode1
        global mode2
        CaptureAgent.registerInitialState(self, gameState)
        self.ignoreCounter = 0
        self.start = gameState.getAgentPosition(self.index)
        self.pastLocation = self.start
        self.depthGrid, self.chokePoints = getDepthsAndChokePoints(gameState)
        
        # 第一个agent会被设置为领导者，第二个agent会被设置为追随者
        if enemyLocationPredictor == None:
            self.isLead = True
            self.targetId = 1
            enemyLocationPredictor = EnemyLocationPredictor(gameState, self)
        else:
            self.targetId = 2
            self.isLead = False

        self.prevFood = self.getFood(gameState)  # 要吃的食物
        self.prevFoodDef = self.getFoodYouAreDefending(gameState)  # 要保护的食物
        self.prevScore = self.getScore(gameState)  # 赢其他队伍的分数
        self.prevCapsules = self.getCapsules(gameState)  # 可以吃到大力丸
        self.prevEnemyCapsules = self.getCapsules(gameState)

        self.openSpaces = 0
        # 计算游戏地图中开放空间的数量
        for a in gameState.getWalls():
            for b in a:
                if not b:
                    self.openSpaces += 1

        # Starting Mode
         # 如果是领导者，mode1为Travel to Center，如果是追随者，mode2为Travel to Center
        self.mode = 'Travel to Center'
        if self.targetId == 1:
            mode1 = self.mode
        else:
            mode2 = self.mode

        self.justSpawned = True

        self.dotsHeld = 0
        self.dotsHeldByInvaders = 0

        teamIndices = self.getTeam(gameState)
        if teamIndices[0] != self.index:
            self.teamIndex = teamIndices[0]
        else:
            self.teamIndex = teamIndices[1]

        self.capsuleOn = False
        self.enemyCapsuleOn = False
        self.timeUntilCapsuleOver = 0
        self.timeUntilEnemyCapsuleOver = 0

        self.target = None

    # Switches between Charge Capsule and Shallow Offense
    def getNextMode(self, gameState, verbose=False):
        """
        根据当前游戏状态确定代理的下一个模式。

        Parameters:
        - gameState (GameState): 当前游戏状态。
        - verbose (bool): 是否打印调试信息。

        Returns:
        - str: 代理的下一个模式

        Important Factors:
        - Last mode
        - Was just captured
        - Invader near
        - Defender near
        - Aggression
        - Score Advantage
        - Whether shallow targets exist
        - Whether there are invaders
        - Whether on Offense
        - Whether Capsule can be charged safely
        - Whether holding enough food
        - is Capsule on
        - is Enemy Capsule on
        """
        global mode1
        global mode2
        global chargeInvaderCounter
        global invaderPresentCounter
        global target1
        global target2
        currentMode = self.mode
        if self.targetId == 1:
            otherMode = mode2
        else:
            otherMode = mode1

        state = gameState.getAgentState(self.index)  # agent的状态
        currentPos = gameState.getAgentPosition(self.index)  # agent的位置
        enemies = [gameState.getAgentState(
            i) for i in self.getOpponents(gameState)]  # 敌人的状态
        enemiesKnown = [a for a in enemies if a.getPosition() !=
                        None]  # 已知敌人的状态
        invaders = [a for a in enemies if a.isPacman]  # 敌人是否是吃豆人
        invadersKnown = [
            a for a in enemies if a.isPacman and a.getPosition() != None]  # 已知敌人是否是吃豆人
        defenders = [a for a in enemies if not a.isPacman]  # 敌人是否是防御者
        defendersKnown = [
            a for a in enemies if not a.isPacman and a.getPosition() != None]  # 已知敌人是否是防御者

        invaderKnownPositions = [a.getPosition()
                                 for a in invadersKnown]  # 已知进攻敌人的位置
        defenderKnownPositions = [a.getPosition()
                                  for a in defendersKnown]  # 已知防守敌人的位置

        # 计算agent到进攻敌人和防守敌人的距离
        defenderDist = 1000
        for pos in defenderKnownPositions:
            dist = self.getMazeDistance(pos, currentPos)
            if dist < defenderDist:
                defenderDist = dist

        invaderDist = 1000
        for pos in invaderKnownPositions:
            dist = self.getMazeDistance(pos, currentPos)
            if dist < invaderDist:
                invaderDist = dist

        # 计算agent的分数差
        scoreDiff = self.getScore(gameState)

        # 计算agent的队友位置
        teamPos = gameState.getAgentPosition(self.teamIndex)
        # 是否存在浅目标
        shallowTargetsExist = False
        # 如果存在浅食物和自己的距离小于浅食物和队友的位置之间的距离，则存在浅目标
        for shallow in getShallowFood(gameState, self, self.red).asList():
            if self.getMazeDistance(shallow, currentPos) < self.getMazeDistance(shallow, teamPos):
                shallowTargetsExist = True
                break

        if len(defendersKnown) > 0:
            capsuleAdvantage = min([getDistAdvantageCapsule(
                gameState, self, a.getPosition(), not state.isPacman) for a in defendersKnown])
        else:
            # 如果没有已知的防守者，则胶囊优势为-1，表示计算没有意义
            capsuleAdvantage = -1

        # TODO
        captured = self.justSpawned  # 代理被抓到了
        invaderNear = invaderDist < 6
        defenderNear = defenderDist < 4
        isAggressive = getAggression(gameState, currentPos, self.red) >= 0.25
        significantAdvantage = scoreDiff > 8  # 分差是否大于8
        invadersExist = len(invaders)
        onOffense = state.isPacman  # 是否处于进攻状态
        canChargeCapsule = capsuleAdvantage >= 0  # 有能力获取胶囊
        holdingEnoughFood = self.dotsHeld >= ENOUGH_DOTS  # 是否持有足够多的食物
        isCapsuleOn = self.capsuleOn
        isEnemyCapsuleOn = self.enemyCapsuleOn  # 是否处于能量豆效果下
        needHelp = chargeInvaderCounter > 30
        aLotOfFoodHeldByEnemies = self.dotsHeldByInvaders > 4
        invaderPresentForAWhile = invaderPresentCounter > 12

        # 当处于冲锋状态
        if currentMode == "Travel to Center":
            # 如果附近有敌人，且敌人没有能量豆效果
            if invaderNear and (not isEnemyCapsuleOn):
                return "Charge Invader"  # 冲锋
            # 如果具有侵略性
            elif isAggressive:
                # 如果分差大于8且处于非进攻状态
                if significantAdvantage and (not onOffense):
                    return "Sentry"  # 守卫
                elif shallowTargetsExist:
                    return "Shallow Offense"  # 浅层进攻
                else:
                    return "Deep Offense"  # 深层进攻
            else:
                return "Travel to Center"  # 走向中心
        # 当处于冲锋状态
        elif currentMode == "Charge Invader":
            # 如果被抓到了
            if captured:
                return "Travel to Center"  # 走向中心
            # 在不是进攻状态的前提下，如果入侵者存在而且附近没有入侵者，或者不存在入侵者但是分差大于8
            elif ((invadersExist and (not invaderNear)) or ((not invadersExist) and significantAdvantage)) and (not onOffense):
                return "Sentry"  # 守卫
            # 入侵者不存在，而且没有得分优势，而且存在浅层食物
            elif (not invadersExist) and (not significantAdvantage) and shallowTargetsExist and (not significantAdvantage):
                return "Shallow Offense"  # 浅层进攻
            # 入侵者不存在，而且没有得分优势，而且没有浅层食物
            elif (not invadersExist) and (not significantAdvantage) and (not shallowTargetsExist) and (not significantAdvantage):
                return "Deep Offense"  # 深层进攻
            else:
                return "Charge Invader"  # 冲锋

        # 当处于撤退状态
        elif currentMode == "Retreat":
            # 如果被抓到了
            if captured:
                return "Travel to Center"
            # 如果附近有入侵者，而且处于进攻状态，而且有能力获取胶囊，而且胶囊没有开启
            elif invaderNear and onOffense and canChargeCapsule and (not isCapsuleOn):
                return "Charge Capsule"  # 向胶囊冲锋
            elif invaderNear and onOffense and canChargeCapsule and (not isCapsuleOn):
                return "Charge Capsule"  # 向胶囊冲锋
            # 如果没有处于进攻状态
            elif not onOffense:
                return "Sentry"  # 守卫
            else:
                return "Retreat"  # 撤退
        # 当处于向胶囊冲锋状态时
        elif currentMode == "Charge Capsule":
            # 如果被抓到了
            if captured:
                return "Travel to Center"  # 走向中心
            # 如果没有能力获取胶囊
            elif (not canChargeCapsule):
                return "Retreat"  # 撤退
            # 如果没有能量获取胶囊且附近有防守者
            elif (not canChargeCapsule) and defenderNear:
                return "Retreat"  # 撤退
            # 如果附近没有防守者而且存在浅层食物，而且没有携带足够食物，而且没有得分优势
            elif (not defenderNear) and shallowTargetsExist and (not holdingEnoughFood) and (not significantAdvantage):
                return "Shallow Offense"  # 浅层进攻
            # 如果附近没有防守者，而且附近没有浅层食物，而且没有携带足够的食物，而且没有得分优势
            elif (not defenderNear) and (not shallowTargetsExist) and (not holdingEnoughFood) and (not significantAdvantage):
                return "Deep Offense"  # 深层进攻
            # 如果胶囊没有开启
            elif not isCapsuleOn:
                return "Charge Capsule"  # 向胶囊冲锋
            else:
                return "Shallow Offense"  # 浅层进攻
        # 当处于拦截入侵者状态时
        elif currentMode == "Intercept Invader":
            # 如果被抓住了
            if captured:
                return "Travel to Center"  # 走向中心
            # 如果附近没有入侵者，而且没有处于进攻状态，而且队友没有在冲锋状态和拦截入侵者状态和守卫状态或者需要帮助
            elif invaderNear and (not onOffense) and ((otherMode != "Charge Invader" and otherMode != "Intercept Invader" and otherMode != "Sentry") or needHelp):
                return "Charge Invader"  # 冲锋状态
            elif defenderNear and (not canChargeCapsule) and (not isCapsuleOn):
                return "Retreat"
            elif defenderNear and canChargeCapsule and (not isCapsuleOn):
                return "Charge Capsule"
            elif (not onOffense) and invadersExist:
                return "Sentry"
            elif (not invaderNear) and (not defenderNear) and shallowTargetsExist and (not holdingEnoughFood) and (not aLotOfFoodHeldByEnemies) and (not invaderPresentForAWhile) and (not significantAdvantage):
                return "Shallow Offense"
            elif (not invaderNear) and (not defenderNear) and (not shallowTargetsExist) and (not holdingEnoughFood) and (not aLotOfFoodHeldByEnemies) and (not invaderPresentForAWhile) and (not significantAdvantage):
                return "Deep Offense"
            else:
                return "Intercept Invader"

        elif currentMode == "Shallow Offense":
            if captured:
                return "Travel to Center"
            elif (defenderNear and (not canChargeCapsule) and (not isCapsuleOn)) or holdingEnoughFood or significantAdvantage:
                return "Retreat"
            elif defenderNear and canChargeCapsule and (not isCapsuleOn):
                return "Charge Capsule"
            elif ((not defenderNear) and invaderNear or aLotOfFoodHeldByEnemies or invaderPresentForAWhile) and ((otherMode != "Charge Invader" and otherMode != "Intercept Invader" and otherMode != "Sentry") or needHelp):
                return "Intercept Invader"
            elif (not defenderNear) and (not shallowTargetsExist) and (not holdingEnoughFood):
                return "Deep Offense"
            else:
                return "Shallow Offense"

        elif currentMode == "Deep Offense":
            if captured:
                return "Travel to Center"
            elif (defenderNear and (not canChargeCapsule) and (not isCapsuleOn)) or holdingEnoughFood or significantAdvantage:
                return "Retreat"
            elif defenderNear and canChargeCapsule and (not isCapsuleOn):
                return "Charge Capsule"
            elif ((not defenderNear) and invaderNear or aLotOfFoodHeldByEnemies or invaderPresentForAWhile) and ((otherMode != "Charge Invader" and otherMode != "Intercept Invader" and otherMode != "Sentry") or needHelp):
                return "Intercept Invader"
            elif (not defenderNear) and shallowTargetsExist and (not holdingEnoughFood):
                return "Shallow Offense"
            else:
                return "Deep Offense"

        elif currentMode == "Sentry":
            if captured:
                return "Travel to Center"
            elif invaderNear:
                return "Charge Invader"
            elif not self.target is None:
                return "Sentry"
            elif (not invadersExist) and (not significantAdvantage) and shallowTargetsExist and (not significantAdvantage):
                return "Shallow Offense"
            elif (not invadersExist) and (not significantAdvantage) and (not shallowTargetsExist) and (not significantAdvantage):
                return "Deep Offense"
            else:
                return "Sentry"

        else:  # Default
            return "Shallow Offense"
        
    def chooseAction(self, gameState):
        global mode1
        global mode2
        global chargeInvaderCounter
        global invaderPresentCounter
        global ignoreCounter1
        global ignoreCounter2
        global random1
        global target1
        global target2

        pos = gameState.getAgentState(self.index).getPosition()
        if self.isLead:
            enemyLocationPredictor.update(
                gameState, self, gameState.getAgentPosition(self.teamIndex), pos)
        investigatePoints = enemyLocationPredictor.getPositionsToInvestigate()
        if pos in investigatePoints:
            enemyLocationPredictor.removePositionFromInvestigation(pos)
        pastMode = self.mode

        self.mode = self.getNextMode(gameState, verbose=False)
        if self.targetId == 1:
            mode1 = self.mode
            otherMode = mode2
        else:
            mode2 = self.mode
            otherMode = mode1

        if self.mode == "Charge Invader" or otherMode == "Charge Invader" and self.isLead:
            chargeInvaderCounter += 1
        elif self.isLead:
            chargeInvaderCounter = 0

        enemies = [gameState.getAgentState(i)
                   for i in self.getOpponents(gameState)]
        enemiesKnown = [a for a in enemies if a.getPosition() != None]
        invaders = [a for a in enemies if a.isPacman]
        if len(invaders) > 0 and self.isLead:
            invaderPresentCounter += 1
        elif self.isLead:
            invaderPresentCounter = 0

        teamPos = gameState.getAgentPosition(self.teamIndex)

        if pos == self.target or teamPos == self.target:
            self.removeTarget(self.targetId)

        if self.isLead:
            if self.mode == "Sentry":
                if len(investigatePoints) > 0:
                    for point in investigatePoints:
                        if self.claimTarget(point, True, 1, pos, teamPos):
                            break
            elif self.mode == "Charge Capsule":
                tempCapsules = self.getCapsules(gameState)
                while len(tempCapsules) > 0:
                    minDist = 100000
                    minTarget = None
                    for point in tempCapsules:
                        dist = self.getMazeDistance(pos, point)
                        if dist < minDist:
                            minDist = dist
                            minTarget = point
                    if not minTarget is None:
                        if self.claimTarget(minTarget, False, 1, pos, teamPos):
                            break
                        else:
                            tempCapsules.remove(minTarget)
                    else:
                        break
            elif self.mode == "Shallow Offense":
                tempFood = getShallowFood(gameState, self, self.red).asList()
                while len(tempFood) > 0:
                    minDist = 100000
                    minTarget = None
                    for point in tempFood:
                        dist = self.getMazeDistance(pos, point)
                        if dist < minDist:
                            minDist = dist
                            minTarget = point
                    if not minTarget is None:
                        if self.getMazeDistance(minTarget, pos) < self.getMazeDistance(minTarget, teamPos):
                            if self.claimTarget(minTarget, False, 1, pos, teamPos):
                                break
                            else:
                                tempFood.remove(minTarget)
                        else:
                            tempFood.remove(minTarget)
                    else:
                        break
                if target1 is None:
                    tempFood = getShallowFood(
                        gameState, self, self.red).asList()
                    minDist = 100000
                    minTarget = None
                    for point in tempFood:
                        dist = self.getMazeDistance(pos, point)
                        if dist < minDist:
                            minDist = dist
                            minTarget = point
                    self.claimTarget(minTarget, False, 1, pos, teamPos, True)
            elif self.mode == "Deep Offense":
                tempFood = getDeepFood(gameState, self, self.red).asList()
                while len(tempFood) > 0:
                    minDist = 100000
                    minTarget = None
                    for point in tempFood:
                        dist = self.getMazeDistance(pos, point)
                        if dist < minDist:
                            minDist = dist
                            minTarget = point
                    if not minTarget is None:
                        if self.claimTarget(minTarget, False, 1, pos, teamPos):
                            break
                        else:
                            tempFood.remove(minTarget)
                    else:
                        break
                if target1 is None:
                    tempFood = getDeepFood(gameState, self, self.red).asList()
                    minDist = 100000
                    minTarget = None
                    for point in tempFood:
                        dist = self.getMazeDistance(pos, point)
                        if dist < minDist:
                            minDist = dist
                            minTarget = point
                    self.claimTarget(minTarget, False, 1, pos, teamPos, True)
            elif self.mode == "Charge Invader":
                self.removeTarget(1)
            # both team's modes
            # team position
            # change claimTarget and removeTarget to require id

            if otherMode == "Sentry":
                if len(investigatePoints) > 0:
                    for point in investigatePoints:
                        if self.claimTarget(point, True, 2, teamPos, pos):
                            break
            elif otherMode == "Charge Capsule":
                tempCapsules = self.getCapsules(gameState)
                while len(tempCapsules) > 0:
                    minDist = 100000
                    minTarget = None
                    for point in tempCapsules:
                        dist = self.getMazeDistance(teamPos, point)
                        if dist < minDist:
                            minDist = dist
                            minTarget = point
                    if not minTarget is None:
                        if self.claimTarget(minTarget, False, 2, teamPos, pos):
                            break
                        else:
                            tempCapsules.remove(minTarget)
                    else:
                        break
            elif otherMode == "Shallow Offense":
                tempFood = getShallowFood(gameState, self, self.red).asList()
                while len(tempFood) > 0:
                    minDist = 100000
                    minTarget = None
                    for point in tempFood:
                        dist = self.getMazeDistance(teamPos, point)
                        if dist < minDist:
                            minDist = dist
                            minTarget = point
                    if not minTarget is None:
                        if self.getMazeDistance(minTarget, pos) > self.getMazeDistance(minTarget, teamPos):
                            if self.claimTarget(minTarget, False, 2, teamPos, pos):
                                break
                            else:
                                tempFood.remove(minTarget)
                        else:
                            tempFood.remove(minTarget)
                    else:
                        break
                if target2 is None:
                    tempFood = getShallowFood(
                        gameState, self, self.red).asList()
                    minDist = 100000
                    minTarget = None
                    for point in tempFood:
                        dist = self.getMazeDistance(teamPos, point)
                        if dist < minDist:
                            minDist = dist
                            minTarget = point
                    self.claimTarget(minTarget, False, 2, teamPos, pos, True)

            elif otherMode == "Deep Offense":
                tempFood = getDeepFood(gameState, self, self.red).asList()
                while len(tempFood) > 0:
                    minDist = 100000
                    minTarget = None
                    for point in tempFood:
                        dist = self.getMazeDistance(teamPos, point)
                        if dist < minDist:
                            minDist = dist
                            minTarget = point
                    if not minTarget is None:
                        if self.claimTarget(minTarget, False, 2, teamPos, pos):
                            break
                        else:
                            tempFood.remove(minTarget)
                    else:
                        break
                if target2 is None:
                    tempFood = getDeepFood(gameState, self, self.red).asList()
                    minDist = 100000
                    minTarget = None
                    for point in tempFood:
                        dist = self.getMazeDistance(teamPos, point)
                        if dist < minDist:
                            minDist = dist
                            minTarget = point
                    self.claimTarget(minTarget, False, 2, teamPos, pos, True)
            elif otherMode == "Charge Invader":
                self.removeTarget(2)

        if self.targetId == 1:
            if ignoreCounter1 == 0:
                self.removeTarget(1)
            elif not self.target is None:
                ignoreCounter1 -= 1
            self.target = target1
        else:
            if ignoreCounter2 == 0:
                self.removeTarget(2)
            elif not self.target is None:
                ignoreCounter2 -= 1
            self.target = target2

        pos = gameState.getAgentState(self.index).getPosition()
        if self.getMazeDistance(pos, self.pastLocation) > 1:
            self.justSpawned = True
        else:
            self.justSpawned = False
        self.pastLocation = pos
        x, y = pos
        x = int(x)
        y = int(y)

        if self.prevFood[x][y]:
            self.dotsHeld += 1
        if (len(self.prevFoodDef.asList()) != len(self.getFoodYouAreDefending(gameState).asList())) and self.prevScore == self.getScore(gameState):
            self.dotsHeldByInvaders += len(self.prevFoodDef.asList()) - len(
                self.getFoodYouAreDefending(gameState).asList())
        elif self.prevScore != self.getScore(gameState):
            self.dotsHeldByInvaders = 0

        if self.prevCapsules != self.getCapsules(gameState):
            self.capsuleOn = True
            self.timeUntilCapsuleOver = 40
        elif self.timeUntilCapsuleOver > 0:
            self.timeUntilCapsuleOver -= 1
        else:
            self.capsuleOn = False

        if self.prevEnemyCapsules != self.getCapsulesYouAreDefending(gameState):
            self.enemyCapsuleOn = True
            self.timeUntilEnemyCapsuleOver = 40
        elif self.timeUntilEnemyCapsuleOver > 0:
            self.timeUntilEnemyCapsuleOver -= 1
        else:
            self.enemyCapsuleOn = False

        self.prevCapsules = self.getCapsules(gameState)
        self.prevEnemyCapsules = self.getCapsulesYouAreDefending(gameState)
        if not gameState.getAgentState(self.index).isPacman:
            self.dotsHeld = 0

        self.prevFood = self.getFood(gameState)
        self.prevFoodDef = self.getFoodYouAreDefending(gameState)
        self.prevScore = self.getScore(gameState)
        actions = gameState.getLegalActions(self.index)
        values = {}

        # Priorities
        priorities = self.getPriorities(self.mode)
        random1 = random.random()
        for action in actions:
            values[action] = self.getFeatures(gameState, self.mode, action)

        for priority in priorities:
            bestVal = -1000000
            goodActions = []
            for action in actions:
                val = values[action][priority]
                if val > bestVal:
                    bestVal = val
                    goodActions = [action]
                elif val == bestVal:
                    goodActions.append(action)
            actions = goodActions
            if len(actions) == 1:
                break

        bestAction = random.choice(goodActions)
        return bestAction

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = gameState.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def getRisk(self, gameState, newPos):
        g = enemyLocationPredictor.getPositionPossibleGrid()
        walls = gameState.getWalls()
        risk = 0.0
        total = 0.0
        for r in range(walls.height):
            for c in range(walls.width):
                pos = (c, r)
                if g[c][r] and not walls[c][r]:
                    if self.getMazeDistance(pos, newPos) <= 5:
                        risk += 1
                        total += 1
                elif not g[c][r] and not walls[c][r]:
                    if self.getMazeDistance(pos, newPos) <= 5:
                        total += 1
        return risk / total

    def getFeatures(self, gameState, mode, action):
        features = dict()
        successor = self.getSuccessor(gameState, action)
        state = successor.getAgentState(self.index)
        pastPos = gameState.getAgentPosition(self.index)
        px, py = pastPos
        teamPos = successor.getAgentPosition(self.teamIndex)
        pos = successor.getAgentPosition(self.index)
        x, y = pos
        investigatePoints = enemyLocationPredictor.getPositionsToInvestigate()
        avoidPoints = enemyLocationPredictor.getPositionsToAvoid()

        isRed = self.red
        enemies = [successor.getAgentState(i)
                   for i in self.getOpponents(gameState)]
        enemiesKnown = [a for a in enemies if a.getPosition() != None]
        invaders = [a for a in enemies if a.isPacman]
        invadersKnown = [
            a for a in enemies if a.isPacman and a.getPosition() != None]
        defenders = [a for a in enemies if not a.isPacman]
        defendersKnown = [
            a for a in enemies if not a.isPacman and a.getPosition() != None]

        invaderKnownPositions = [a.getPosition() for a in invadersKnown]
        defenderKnownPositions = [a.getPosition() for a in defendersKnown]
        enemyKnownPositions = invaderKnownPositions + defenderKnownPositions

        minDistEnemy = 1000
        if len(enemiesKnown) > 0:
            for enemy in enemyKnownPositions:
                dist = self.getMazeDistance(pos, enemy)
                if dist < minDistEnemy:
                    minDistEnemy = dist

        if self.red:
            scoreDiff = self.getScore(gameState)
        else:
            scoreDiff = self.getScore(gameState) * -1

        temp = avoidPoints + defenderKnownPositions
        if len(temp) == 0:
            minDistToDefender = 1000
        else:
            minDistToDefender = 1000
            for point in temp:
                dist = self.getMazeDistance(pos, point)
                if dist < minDistToDefender:
                    minDistToDefender = dist

        temp = investigatePoints + invaderKnownPositions
        if len(temp) == 0:
            minDistToInvader = 1000
        else:
            minDistToInvader = 1000
            for point in temp:
                dist = self.getMazeDistance(pos, point)
                if dist < minDistToInvader:
                    minDistToInvader = dist

        shallowFood = getShallowFood(successor, self, isRed).asList()
        deepFood = getDeepFood(successor, self, isRed).asList()

        # Higher values are preferred

        if self.getMazeDistance(pos, pastPos) > 1 or (minDistEnemy < 2 and self.enemyCapsuleOn) or (minDistToDefender < 2 and not self.capsuleOn):
            features['captured'] = 0
        else:
            features['captured'] = 1
        if action == 'Stop':
            features['stop'] = 0
        else:
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(
            self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 0
        else:
            features['reverse'] = 1

        features['chokeDepth'] = -1 * self.depthGrid[x][y]
        features['aggression'] = -1 * getDistToSafety(successor, self)
        features['distToInvader'] = -1 * minDistToInvader
        features['distToDefender'] = minDistToDefender
        features['invaders'] = -1 * len(invaders)
        features['distToSafety'] = -1 * getDistToSafety(successor, self)
        features['capsuleDist'] = -1 * \
            getMinDistToCapsule(successor, self, pos,
                                self.getCapsules(successor))
        features['foodEaten'] = -1 * len(self.getFood(successor).asList())
        if len(deepFood) == 0:
            features['distToDeepFood'] = 0
        else:
            features['distToDeepFood'] = -1 * \
                min([self.getMazeDistance(pos, food) for food in deepFood])
        if len(shallowFood) == 0:
            features['distToShallowFood'] = 0
        else:
            features['distToShallowFood'] = -1 * \
                min([self.getMazeDistance(pos, food) for food in shallowFood])

        if self.target is None:
            features['teamDist'] = 0
        else:
            features['teamDist'] = self.getMazeDistance(pos, teamPos)

        features['enemyPredictOffense'] = -1 * self.getRisk(gameState, pos)
        features['enemyPredictDefense'] = -1 * features['enemyPredictOffense']

        if self.target is None:
            features['targetDist'] = 0
        else:
            features['targetDist'] = -1 * \
                self.getMazeDistance(pos, self.target)

        if state.isPacman:
            features['onOffense'] = 1
            features['onDefense'] = 1
        else:
            features['onOffense'] = 0
            features['onDefense'] = 1

        if len(self.getCapsulesYouAreDefending(successor)) > 0:
            features['capsuleDist'] = -1 * sum([self.getMazeDistance(
                pos, capsule) for capsule in self.getCapsulesYouAreDefending(successor)])
        else:
            features['capsuleDist'] = 0

        if len(self.getCapsules(successor)) > 0:
            features['capsuleDistOffense'] = -1 * min([self.getMazeDistance(
                pos, capsule) for capsule in self.getCapsules(successor)])
        else:
            features['capsuleDistOffense'] = 0

        features['sentryScore'] = (features['capsuleDist'] * random1) + (
            features['distToSafety'] * (1-random1)) + features['teamDist']
        return features

    def getPriorities(self, mode):
        if mode == "Travel to Center":
            return ['captured', 'stop', 'distToSafety', 'aggression', 'chokeDepth']
        elif mode == "Charge Invader":
            return ['captured', 'stop', 'onDefense', 'invaders', 'distToInvader', 'teamDist', 'reverse']
        elif mode == "Retreat":
            return ['captured', 'stop', 'onDefense', 'distToSafety', 'distToDefender', 'enemyPredictOffense', 'reverse']
        elif mode == "Charge Capsule":
            return ['captured', 'stop', 'targetDist', 'capsuleDistOffense', 'reverse']
        elif mode == "Intercept Invader":
            return ['captured', 'stop', 'distToSafety', 'distToInvader', 'teamDist', 'reverse']
        elif mode == 'Shallow Offense':
            return ['captured', 'stop', 'onOffense', 'targetDist', 'foodEaten', 'teamDist', 'distToShallowFood', 'enemyPredictOffense']
        elif mode == 'Deep Offense':
            return ['captured', 'stop', 'onOffense', 'targetDist', 'foodEaten', 'teamDist', 'distToDeepFood', 'enemyPredictOffense']
        elif mode == 'Sentry':
            return ['captured', 'stop', 'onDefense', 'targetDist', 'sentryScore', 'teamDist', 'distToInvader', 'enemyPredictDefense']
        else:
            return []

    def claimTarget(self, target, investigate, targetId, pos, otherPos, override=False):
        global target1
        global target2
        global ignoreCounter1
        global ignoreCounter2
        if target == None:
            return False
        if targetId == 1:
            if not target1 is None:
                return True
        if target2 == 2:
            if not target2 is None:
                return True

        if targetId == 1:
            if self.getMazeDistance(target, pos) < self.getMazeDistance(target, otherPos) or override:
                if investigate:
                    enemyLocationPredictor.removePositionFromInvestigation(
                        target)
                target1 = target
                ignoreCounter1 = 12
                return True
        else:
            if self.getMazeDistance(target, pos) < self.getMazeDistance(target, otherPos) or override:
                if investigate:
                    enemyLocationPredictor.removePositionFromInvestigation(
                        target)
                target2 = target
                ignoreCounter2 = 12
                return True
        return False

    def removeTarget(self, targetId):
        global target1
        global target2
        global random1
        if targetId == 1:
            target1 = None
            ignoreCounter1 = -1
        else:
            target2 = None
            ignoreCounter2 = -1


