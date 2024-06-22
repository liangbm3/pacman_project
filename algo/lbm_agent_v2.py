# myTeam.py
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
import math

#################
# Team Pac-Champs #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########



class ReflexCaptureAgent(CaptureAgent):
    """
    得分最大化的基类
    """
    def getSuccessor(self, gameState, action):
        """
        返回采取给定动作的后续游戏状态。
        Args:
            gameState (GameState): 游戏状态
            action (str): 采取的动作

        Returns:
            GameState: 后续游戏状态
        """
        #获取下一步的游戏状态
        successor = gameState.generateSuccessor(self.index, action)
        #下一步的位置
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        评估给定的游戏状态中的给定动作。

        Args:
            gameState (GameState): 游戏状态
            action (str): 被评估的动作

        Returns:
            float: 该动作的评估分数。
        """
        features = self.evaluateAttackParameters(gameState, action)
        weights = self.getCostOfAttackParameter(gameState, action)
        return features * weights

    def evaluateAttackParameters(self, gameState, action):
        """
        评估给定游戏状态下给定动作的攻击参数。

        Args:
            gameState (GameState): 要评估的游戏状态
            action (str): 要评估的动作

        Returns:
            Counter: 包含评估的攻击参数的计数器对象。
        """
        features = util.Counter()#类似于python的字典，用于计数，初始默认值为0
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)#我们的得分和对手的得分的差值
        return features

    def getCostOfAttackParameter(self, gameState, action):
        """
        返回给定游戏状态下给定动作的攻击参数的成本。

        Args:
            gameState (GameState): 要评估的游戏状态
            action (str): 要评估的动作

        Returns:
            dict: 包含攻击参数成本的字典。
        """
        return {'successorScore': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    一个进攻性agent
    """
    def __init__(self, index):
        """
        Parameters:
        - index (int): agent的索引

        Attributes:
        - presentCoordinates (tuple): agent的当前坐标。
        - counter (int): 计数器变量。
        - attack (bool): 指示代理是否处于攻击模式。
        - lastFood (list): 最后已知的食物位置的坐标列表。
        - presentFoodList (list): 当前食物位置的坐标列表。
        - shouldReturn (bool): 指示代理是否应该返回其所在一侧。
        - capsulePower (bool): 表示agent是否有能力吃掉胶囊。
        - targetMode (None or str): agent目标的模式。
        - eatenFood (int): agent吃掉的食物数量。
        - initialTarget (list): agent的初始目标坐标。
        - hasStopped (int): 指示agent停止的次数的计数器变量。
        - capsuleLeft (int): 游戏中剩余的胶囊数量。
        - prevCapsuleLeft (int): 上一个游戏状态下剩余的胶囊数量。
        """
        CaptureAgent.__init__(self, index)        
        self.presentCoordinates = (-5 ,-5)
        self.counter = 0
        self.attack = False
        self.lastFood = []
        self.presentFoodList = []
        self.shouldReturn = False
        self.capsulePower = False
        self.targetMode = None
        self.eatenFood = 0
        self.initialTarget = []
        self.hasStopped = 0
        self.capsuleLeft = 0
        self.prevCapsuleLeft = 0

    def registerInitialState(self, gameState):
        """
        在游戏开始时初始化agent的状态。

        Args:
            gameState (GameState): 目前的游戏状态

        Returns:
            None
        """
        self.currentFoodSize = 9999999 #初始食物数量
        CaptureAgent.registerInitialState(self, gameState)  
        self.initPosition = gameState.getAgentState(self.index).getPosition()#agent的初始位置
        self.initialAttackCoordinates(gameState)#确定agent的初始攻击坐标

    def initialAttackCoordinates(self, gameState):
        """
        确定agent的初始攻击坐标。

        Args:
            gameState (object): 目前的游戏状态

        Returns:
            None

        """
        
        #获取游戏地图的宽度和高度，并初始化相应大小的矩阵
        layoutInfo = [] 
        x = (gameState.data.layout.width - 2) // 2
        if not self.red:
            x += 1
        y = (gameState.data.layout.height - 2) // 2
        layoutInfo.extend((gameState.data.layout.width, gameState.data.layout.height, x, y))

        self.initialTarget = []

        #遍历地图每一行，如果没有墙，则将坐标添加到初始目标列表中
        for i in range(1, layoutInfo[1] - 1):
            if not gameState.hasWall(layoutInfo[2], i):
                self.initialTarget.append((layoutInfo[2], i))

        #如果初始目标列表的长度为偶数，则设为中间值，否则设为中间值的前一个值
        noTargets = len(self.initialTarget)
        if noTargets % 2 == 0:
            noTargets = noTargets // 2
            self.initialTarget = [self.initialTarget[noTargets]]
        else:
            noTargets = (noTargets - 1) // 2
            self.initialTarget = [self.initialTarget[noTargets]]

    
    def evaluateAttackParameters(self, gameState, action):
        """
        评估agent的攻击参数。这里继承并重写了ReflexCaptureAgent类的evaluateAttackParameters方法。

        Args:
            gameState (GameState): 要评估的游戏状态。
            action (str): 要评估的动作。

        Returns:
            Counter: 包含评估的攻击参数的计数器对象。
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action) 
        position = successor.getAgentState(self.index).getPosition() #获取下一状态agent的位置
        foodList = self.getFood(successor).asList() #获取下一状态的食物列表
        features['successorScore'] = self.getScore(successor) #获取下一状态的得分

        #如果agent是吃豆人，则offence为1，代表正在敌方领域，否则为0，代表在己方领域
        if successor.getAgentState(self.index).isPacman:
            features['offence'] = 1
        else:
            features['offence'] = 0

        if foodList: 
            features['foodDistance'] = min([self.getMazeDistance(position, food) for food in foodList])#计算到最近食物的距离

        #存储对手信息，是一个对手的索引列表
        opponentsList = []
       
        #和鬼的距离
        disToGhost = []
        opponentsList = self.getOpponents(successor)

        #如果对手是鬼，且位置不为空，则计算到鬼的距离
        for i in range(len(opponentsList)):
            enemyPos = opponentsList[i]
            enemy = successor.getAgentState(enemyPos)
            if not enemy.isPacman and enemy.getPosition() != None:
                ghostPos = enemy.getPosition()
                disToGhost.append(self.getMazeDistance(position ,ghostPos))

        #如果到鬼的距离列表不为空，则取最小值
        if len(disToGhost) > 0:
            minDisToGhost = min(disToGhost)
            if minDisToGhost < 5:
                features['distanceToGhost'] = minDisToGhost + features['successorScore']
            else:
                features['distanceToGhost'] = 0


        return features
    
    def getCostOfAttackParameter(self, gameState, action):
        '''
        根据游戏状态和动作返回攻击参数的成本。这里继承并重写了ReflexCaptureAgent类的getCostOfAttackParameter方法。

        Parameters:
            gameState (GameState): 要评估的游戏状态。
            action (str): 要评估的动作。

        Returns:
            dict: 包含攻击参数成本的字典。

        经过多次迭代后手动设置权重。
        '''
        
        #手动设置权重
        
        #进行攻击的权重
        if self.attack:
            if self.shouldReturn is True:
                return {'offence' :3010,
                        'successorScore': 202,
                        'foodDistance': -8,
                        'distancesToGhost' :215}
            else:
                return {'offence' :0,
                        'successorScore': 202,
                        'foodDistance': -8,
                        'distancesToGhost' :215}
        #不进行攻击的权重
        else:
            successor = self.getSuccessor(gameState, action) 
            weightGhost = 210
            #获取敌方游戏状态
            enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
            #如果敌人不是吃豆人且位置不为空，则计算到敌人的距离，这时候附近有敌人
            invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
            if len(invaders) > 0:
                if invaders[-1].scaredTimer > 0:#如果我们还有恐吓时间，则不用担心
                    weightGhost = 0
                    
            return {'offence' :0,
                    'successorScore': 202,
                    'foodDistance': -8,
                    'distancesToGhost' :weightGhost}

    def getOpponentPositions(self, gameState):
        """
        获取所有对手的位置。

        Parameters:
        - gameState: 当前游戏状态

        Returns:
        - list: 所有对手的位置
        """
        return [gameState.getAgentPosition(enemy) for enemy in self.getOpponents(gameState)]

    def bestPossibleAction(self, mcsc):
        """
        获取agent的最佳可能动作。

        Parameters:
        - mcsc (MonteCarloTreeSearch): 蒙特卡洛树搜索对象

        Returns:
        - str: 最佳可能动作
        """
        ab = mcsc.getLegalActions(self.index)
        #移除停止动作
        ab.remove(Directions.STOP)

        #如果只有一个动作，则返回该动作，没有其他选择
        if len(ab) == 1:
            return ab[0]
        
        #否则，返回一个随机选择的动作
        else:
            #获取agent的当前方向的反方向，如果在动作列表中，则移除，避免返回反方向
            reverseDir = Directions.REVERSE[mcsc.getAgentState(self.index).configuration.direction]
            if reverseDir in ab:
                ab.remove(reverseDir)
            return random.choice(ab)

    def monteCarloSimulation(self, gameState, depth):
        """
        蒙特卡洛模拟。

        Args:
            gameState (GameState): 当前游戏状态
            depth (int): 深度

        Returns:
            float: 评估分数
        """
        
        #创建深度拷贝
        ss = gameState.deepCopy()
        #根据深度进行模拟，深度决定模拟的次数
        while depth > 0:
            ss = ss.generateSuccessor(self.index, self.bestPossibleAction(ss))
            depth -= 1
        return self.evaluate(ss, Directions.STOP)#将最后的游戏状态和stop动作传入评估函数，并返回评估分数

    def getBestAction(self, legalActions, gameState, possibleActions, distanceToTarget):
        """
        根据给定的合法行动、游戏状态、可能行动以及与目标的距离返回最佳行动。

        Parameters:
        - legalActions (list): 一个包含合法行动的列表。
        - gameState: 当前游戏状态。
        - possibleActions: 一个列表，用于存储agent的可能行动。
        - distanceToTarget (list): 一个列表，用于存储agent到目标的距离。

        Returns:
        - bestAction: 最佳动作
        """
        
        #初始化最短距离
        shortestDistance = 9999999999
        
        #遍历合法行动，计算到目标的距离，找到最短距离
        for i in range(0, len(legalActions)):
            action = legalActions[i]
            nextState = gameState.generateSuccessor(self.index, action)
            nextPosition = nextState.getAgentPosition(self.index)
            distance = self.getMazeDistance(nextPosition, self.initialTarget[0])
            distanceToTarget.append(distance)
            if distance < shortestDistance:
                shortestDistance = distance

        #列表包含agent到目标的最短距离的所有动作
        bestActionsList = [a for a, distance in zip(legalActions, distanceToTarget) if distance == shortestDistance]
        bestAction = random.choice(bestActionsList)#随机选择一个最佳动作
        return bestAction
        
    def chooseAction(self, gameState):
        
        #当前坐标
        self.presentCoordinates = gameState.getAgentState(self.index).getPosition()
    
        if self.presentCoordinates == self.initPosition:
            self.hasStopped = 1
        if self.presentCoordinates == self.initialTarget[0]:
            self.hasStopped = 0

        #如果agent停止，则返回最佳动作
        if self.hasStopped == 1:
            legalActions = gameState.getLegalActions(self.index)
            legalActions.remove(Directions.STOP)
            possibleActions = []
            distanceToTarget = []
            
            bestAction=self.getBestAction(legalActions,gameState,possibleActions,distanceToTarget)
            
            return bestAction
        #否则，进行攻击
        if self.hasStopped==0:
            self.presentFoodList = self.getFood(gameState).asList()
            self.capsuleLeft = len(self.getCapsules(gameState))
            realLastCapsuleLen = self.prevCapsuleLeft
            realLastFoodLen = len(self.lastFood)

            # 当吃豆人获得一些食物并应该返回家时，设置 returned = 1          
            if (len(self.lastFood) - len(self.presentFoodList)) > 6:
                self.shouldReturn = True
            self.lastFood = self.presentFoodList
            self.prevCapsuleLeft = self.capsuleLeft

           #如果不是pacman，则shouldReturn为False
            if not gameState.getAgentState(self.index).isPacman:
                self.shouldReturn = False

            # 检查攻击情况           
            remainingFoodList = self.getFood(gameState).asList()#剩余食物列表
            remainingFoodSize = len(remainingFoodList)#剩余食物数量
    
            #如果剩余食物数量和当前食物数量相等，则计数器加1，否则重置计数器
            if remainingFoodSize == self.currentFoodSize:
                self.counter = self.counter + 1
            else:
                self.currentFoodSize = remainingFoodSize
                self.counter = 0
                
            #如果agent的初始位置和当前位置相同，则计数器重置为0，如果死亡了会进入这个判断
            if gameState.getInitialAgentPosition(self.index) == gameState.getAgentState(self.index).getPosition():
                self.counter = 0
                
            #如果计数器大于20，则进入攻击模式，否则为False
            if self.counter > 20:
                self.attack = True
            else:
                self.attack = False
            
            #获取可执行的所有动作，并移除停止动作
            actionsBase = gameState.getLegalActions(self.index)
            actionsBase.remove(Directions.STOP)

            # 找出和最近敌人的距离       
            distanceToEnemy = 999999
            enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
            invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer == 0]
            if len(invaders) > 0:
                distanceToEnemy = min([self.getMazeDistance(self.presentCoordinates, a.getPosition()) for a in invaders])
            
            '''
            吃胶囊：-> 如果有胶囊可用，则 capsulePower 为 True。
            -> 如果敌人距离小于 5，则 capsulePower 为 False。
            -> 如果吃豆人获得食物，则返回家中 capsulePower 为 False.
            '''
            if self.capsuleLeft < realLastCapsuleLen:
                self.capsulePower = True
                self.eatenFood = 0
            if distanceToEnemy <= 5:
                self.capsulePower = False
            if (len (self.lastFood)-len(self.presentFoodList) > 4 ):
                self.capsulePower = False

        
            if self.capsulePower:
                if not gameState.getAgentState(self.index).isPacman:
                    self.eatenFood = 0

                modeMinDistance = 999999

                if len(self.presentFoodList) < realLastFoodLen:
                    self.eatenFood += 1

                if len(self.presentFoodList )==0 or self.eatenFood >= 5:
                    self.targetMode = self.initPosition
        
                else:
                    for food in self.presentFoodList:
                        distance = self.getMazeDistance(self.presentCoordinates ,food)
                        if distance < modeMinDistance:
                            modeMinDistance = distance
                            self.targetMode = food

                legalActions = gameState.getLegalActions(self.index)
                legalActions.remove(Directions.STOP)
                possibleActions = []
                distanceToTarget = []
                
                k=0
                while k!=len(legalActions):
                    a = legalActions[k]
                    newpos = (gameState.generateSuccessor(self.index, a)).getAgentPosition(self.index)
                    possibleActions.append(a)
                    distanceToTarget.append(self.getMazeDistance(newpos, self.targetMode))
                    k+=1
                
                minDis = min(distanceToTarget)
                bestActions = [a for a, dis in zip(possibleActions, distanceToTarget) if dis== minDis]
                bestAction = random.choice(bestActions)
                return bestAction
            else:
               
                self.eatenFood = 0
                distanceToTarget = []
                for a in actionsBase:
                    nextState = gameState.generateSuccessor(self.index, a)
                    value = 0
                    for i in range(1, 24):
                        value += self.monteCarloSimulation(nextState ,20)
                    distanceToTarget.append(value)

                best = max(distanceToTarget)
                bestActions = [a for a, v in zip(actionsBase, distanceToTarget) if v == best]
                bestAction = random.choice(bestActions)
            return bestAction


class DefensiveReflexAgent(ReflexCaptureAgent):
    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.target = None
        self.previousFood = []
        self.counter = 0

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.setPatrolPoint(gameState)

    def setPatrolPoint(self ,gameState):
        '''
        Look for center of the maze for patrolling
        '''
        x = (gameState.data.layout.width - 2) // 2
        if not self.red:
            x += 1
        self.patrolPoints = []
        for i in range(1, gameState.data.layout.height - 1):
            if not gameState.hasWall(x, i):
                self.patrolPoints.append((x, i))

        for i in range(len(self.patrolPoints)):
            if len(self.patrolPoints) > 2:
                self.patrolPoints.remove(self.patrolPoints[0])
                self.patrolPoints.remove(self.patrolPoints[-1])
            else:
                break
    

    def getNextDefensiveMove(self ,gameState):

        agentActions = []
        actions = gameState.getLegalActions(self.index)
        
        rev_dir = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        actions.remove(Directions.STOP)

        for i in range(0, len(actions)-1):
            if rev_dir == actions[i]:
                actions.remove(rev_dir)


        for i in range(len(actions)):
            a = actions[i]
            new_state = gameState.generateSuccessor(self.index, a)
            if not new_state.getAgentState(self.index).isPacman:
                agentActions.append(a)
        
        if len(agentActions) == 0:
            self.counter = 0
        else:
            self.counter = self.counter + 1
        if self.counter > 4 or self.counter == 0:
            agentActions.append(rev_dir)

        return agentActions

    def chooseAction(self, gameState):
        
        position = gameState.getAgentPosition(self.index)
        if position == self.target:
            self.target = None
        invaders = []
        nearestInvader = []
        minDistance = float("inf")


        # Look for enemy position in our home        
        opponentsPositions = self.getOpponents(gameState)
        i = 0
        while i != len(opponentsPositions):
            opponentPos = opponentsPositions[i]
            opponent = gameState.getAgentState(opponentPos)
            if opponent.isPacman and opponent.getPosition() != None:
                opponentPos = opponent.getPosition()
                invaders.append(opponentPos)
            i = i + 1

        # if enemy is found chase it and kill it
        if len(invaders) > 0:
            for oppPosition in invaders:
                dist = self.getMazeDistance(oppPosition ,position)
                if dist < minDistance:
                    minDistance = dist
                    nearestInvader.append(oppPosition)
            self.target = nearestInvader[-1]

        # if enemy has eaten some food, then remove it from targets
        else:
            if len(self.previousFood) > 0:
                if len(self.getFoodYouAreDefending(gameState).asList()) < len(self.previousFood):
                    yummy = set(self.previousFood) - set(self.getFoodYouAreDefending(gameState).asList())
                    self.target = yummy.pop()

        self.previousFood = self.getFoodYouAreDefending(gameState).asList()
        
        if self.target == None:
            if len(self.getFoodYouAreDefending(gameState).asList()) <= 4:
                highPriorityFood = self.getFoodYouAreDefending(gameState).asList() + self.getCapsulesYouAreDefending(gameState)
                self.target = random.choice(highPriorityFood)
            else:
                self.target = random.choice(self.patrolPoints)
        candAct = self.getNextDefensiveMove(gameState)
        awsomeMoves = []
        fvalues = []

        i=0
        
        # find the best move       
        while i < len(candAct):
            a = candAct[i]
            nextState = gameState.generateSuccessor(self.index, a)
            newpos = nextState.getAgentPosition(self.index)
            awsomeMoves.append(a)
            fvalues.append(self.getMazeDistance(newpos, self.target))
            i = i + 1

        best = min(fvalues)
        bestActions = [a for a, v in zip(awsomeMoves, fvalues) if v == best]
        bestAction = random.choice(bestActions)
        return bestAction