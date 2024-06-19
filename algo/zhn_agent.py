# myTeam.py

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util
from game import Directions
import game
from util import nearestPoint

# 创建团队函数，返回两个代理，分别是进攻型代理和防守型代理
def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveAgent', second = 'DefensiveAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

# 进攻型代理类
class OffensiveAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        # 记录初始位置
        self.start = gameState.getAgentPosition(self.index)
        # 调用父类方法进行初始化
        CaptureAgent.registerInitialState(self, gameState)
        # 初始化一个变量来记录吃掉的食物数量
        self.foodCarried = 0
    
    def chooseAction(self, gameState):
        
        # 获取所有合法动作
        actions = gameState.getLegalActions(self.index)
        # 评估每个动作的得分
        values = [self.evaluate(gameState, a) for a in actions] 
        # 找到最高得分
        maxValue = max(values)
        # 选择得分最高的动作
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        # 获取剩余食物的数量
        foodLeft = len(self.getFood(gameState).asList())

        # 获取当前代理的位置
        myPos = gameState.getAgentState(self.index).getPosition()

        # 获取敌人的位置和状态
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        # 入侵者
        invaders= [a for a in enemies if  a.isPacman and a.getPosition() != None]
        # 鬼
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        scaredGhosts = [a for a in invaders if a.scaredTimer > 0]
        activeGhosts = [a for a in invaders if a.scaredTimer == 0]

        # 获取胶囊的位置
        capsules = self.getCapsules(gameState)

        Changepos = self.getpreLocation_GhostToPacman(gameState)
        # 如果吃掉的食物数量大于等于15或剩余食物少于等于2个或游戏时间只剩下回家的时间，优先返回起始位置
        if self.foodCarried >= 15 or foodLeft <= 2 or gameState.data.timeleft < self.getMazeDistance(myPos, self.start) + 100:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            # 如果已经返回到起始位置，重置吃掉的食物数量
            if bestDist == 0:
                self.foodCarried = 0
            return bestAction

        # 如果场上没有安全的食物，场上还有胶囊，对手处于恐吓剩下的时间不多时，进入搜索胶囊状态
        if len(self.getSafeFood(gameState)) == 0 and len(capsules) > 0 and any([ghost.scaredTimer < 10 for ghost in scaredGhosts]):
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = min([self.getMazeDistance(pos2, capsule) for capsule in capsules])
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        # 如果pacman身上没有携带食物且场上存在安全的食物，进入食用安全的食物状态
        if self.foodCarried == 0 and len(self.getSafeFood(gameState)) > 0:
            bestDist = 9999
            safeFood = self.getSafeFood(gameState)
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = min([self.getMazeDistance(pos2, food) for food in safeFood])
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        # 如果pacman身上没有携带食物且场上没有安全的食物，直接去搜索食物
        if self.foodCarried == 0 and len(self.getSafeFood(gameState)) == 0:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = min([self.getMazeDistance(pos2, food) for food in self.getFood(gameState).asList()])
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        # 如果鬼出现且与pacman相距不远，进入逃生状态
        if len(activeGhosts) > 0 and min([self.getMazeDistance(myPos, ghost.getPosition()) for ghost in activeGhosts]) < 5:
            bestDist = -9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = min([self.getMazeDistance(pos2, ghost.getPosition()) for ghost in activeGhosts])
                if dist > bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        # 如果大力丸的持续时间足够，进入搜索危险食物状态
        if len(scaredGhosts) > 0 and max([ghost.scaredTimer for ghost in scaredGhosts]) > 20:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = min([self.getMazeDistance(pos2, food) for food in self.getFood(gameState).asList()])
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        # 否则，从得分最高的动作中随机选择一个
        return random.choice(bestActions)
    
    def getpreLocation_GhostToPacman(self,gameState):
        """当agent从ghost变为pacman时，获取上一个状态的位置"""
        prevObservation = self.getPreviousObservation()
        if prevObservation:
            # 检查代理在上一个状态和当前状态是否是Pacman
            wasPacman = prevObservation.getAgentState(self.index).isPacman
            isPacman = gameState.getAgentState(self.index).isPacman

            # 当上一个状态不是Pacman，而当前状态是Pacman时，获取上一个状态的位置
            if not wasPacman and isPacman:
                # 获取上一个状态时代理的位置
                prevPos = prevObservation.getAgentState(self.index).getPosition()
                return prevPos
        return None
    def getSuccessor(self, gameState, action):
        # 获取当前动作的后继状态
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        # 如果位置不是整数点，再次生成后继状态
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        # 计算特征值和特征权重的线性组合
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        # 获取当前动作的特征
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()    
        features['successorScore'] = -len(foodList)

        # 计算到最近食物的距离
        if len(foodList) > 0:
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        # 计算到敌方入侵者的距离
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        # 如果吃掉了食物，增加吃掉的食物数量
        if action in gameState.getLegalActions(self.index):
            nextState = gameState.generateSuccessor(self.index, action)
            nextFoodList = self.getFood(nextState).asList()
            if len(nextFoodList) < len(foodList):
                self.foodCarried += 1

        return features

    def getWeights(self, gameState, action):
        # 定义特征的权重
        return {'successorScore': 100, 'distanceToFood': -1, 'invaderDistance': -10}

    def getSafeFood(self, gameState):
        # 获取安全的食物，即没有鬼靠近的食物
        foodList = self.getFood(gameState).asList()
        myPos = gameState.getAgentState(self.index).getPosition()
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        safeFood = []
        # 遍历所有食物，检查是否有鬼靠近
        for food in foodList:
            safe = True
            for ghost in ghosts:
                if self.getMazeDistance(food, ghost.getPosition()) < 5:
                    safe = False
                    break
            if safe:
                safeFood.append(food)
        return safeFood


# 防守型代理类
class DefensiveAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        # 记录初始位置
        self.start = gameState.getAgentPosition(self.index)
        # 调用父类方法进行初始化
        CaptureAgent.registerInitialState(self, gameState)
    
    def chooseAction(self, gameState):
        # 获取所有合法动作
        actions = gameState.getLegalActions(self.index)
        # 评估每个动作的得分
        values = [self.evaluate(gameState, a) for a in actions]
        # 找到最高得分
        maxValue = max(values)
        # 选择得分最高的动作
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        # 从得分最高的动作中随机选择一个
        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        # 获取当前动作的后继状态
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        # 如果位置不是整数点，再次生成后继状态
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        # 计算特征值和特征权重的线性组合
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        # 获取当前动作的特征
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # 计算是否在防守状态
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # 计算到敌方入侵者的距离
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        # 如果动作是停止或者反方向，增加惩罚
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        # 定义特征的权重
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}



