# myTeam.py

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util
from game import Directions
import game
from game import Agent
from capture import SIGHT_RANGE
# 创建团队函数，返回两个代理，分别是进攻型代理和防守型代理
def createTeam(firstIndex, secondIndex, isRed,
               first = 'CautiousAttackAgent', second = 'DefensiveAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


class ApproximateAdversarialAgent(CaptureAgent):
    """
        动作选取通过alpha-beta 剪枝,看不见的敌人用贝叶斯推理近似
    """


    SEARCH_DEPTH = 2  # 搜索深度

    def registerInitialState(self, gameState):
            CaptureAgent.registerInitialState(self, gameState)

            # 获取棋盘上所有非墙壁位置
            self.legalPositions = gameState.data.layout.walls.asList(False)

            # 初始化对手的位置信念分布
            self.positionBeliefs = {}
            for opponent in self.getOpponents(gameState):   # 遍历所有对手的索引
                self.initializeBeliefs(opponent)  # 初始化对手的位置信念分布

    def initializeBeliefs(self, agent):
        """
    设置敌人初始信念分布为1.0
        """
        self.positionBeliefs[agent] = util.Counter()        
        for p in self.legalPositions:        # 遍历所有非墙壁位置
            self.positionBeliefs[agent][p] = 1.0  # 初始信念分布为1.0

    def chooseAction(self, gameState):
        # 更新关于对手位置的信念分布，并将隐藏的对手放置在他们最可能的位置
        noisyDistances = gameState.getAgentDistances()  # 获取所有agent的噪声距离
        probableState = gameState.deepCopy()    # 复制当前状态

        for opponent in self.getOpponents(gameState):   # 遍历所有对手的索引
            pos = gameState.getAgentPosition(opponent)    # 获取对手位置
            if pos:   # 如果对手被识别到
                self.fixPosition(opponent, pos)  # 对手位置信念分布设置为固定值
            else:  # 如果对手未被识别到
                self.elapseTime(opponent, gameState)    # 假设对手随机移动，但也要检查前一回合中丢失的食物,更新对手位置信念分布
                self.observe(opponent, noisyDistances[opponent], gameState)     # 再根据噪声距离更新对手位置信念分布     

        #self.displayDistributionsOverPositions(self.positionBeliefs.values())  # 更新信念分布图  
        for opponent in self.getOpponents(gameState):   # 遍历所有对手的索引
            probablePosition = self.guessPosition(opponent)    # 猜测对手位置

            # 构造对手位置的配置
            conf = game.Configuration(probablePosition, Directions.STOP)      
            probableState.data.agentStates[opponent] = game.AgentState(conf, probableState.isRed(probablePosition) != probableState.isOnRedTeam(opponent))

        #运行negamax算法与alpha-beta剪枝来选择一个最优的移动
        bestVal, bestAction = float("-inf"), None
        for opponent in self.getOpponents(gameState):
            value, action = self.expectinegamax(opponent,probableState,self.SEARCH_DEPTH,1, retAction=True)
            if value > bestVal:
                    bestVal, bestAction = value, action

        return action

    def fixPosition(self, agent, position):
        """
        将对手的位置信念分布设置为固定值
        """
        updatedBeliefs = util.Counter()
        updatedBeliefs[position] = 1.0
        self.positionBeliefs[agent] = updatedBeliefs

    def elapseTime(self, agent, gameState):
        """
        假设对手随机移动，但也要检查前一回合中丢失的食物。
        """
        updatedBeliefs = util.Counter()  # 新建信念分布
        for (oldX, oldY), oldProbability in self.positionBeliefs[agent].items():  # 遍历信念分布得到每个位置及其信念分布
            newDist = util.Counter()
            for pos in [(oldX - 1, oldY), (oldX + 1, oldY),(oldX, oldY - 1), (oldX, oldY + 1)]: # 随机移动的位置
                if pos in self.legalPositions:    # 如果随机移动的pos合法
                    newDist[pos] = 1.0    # 信念分布仍然为1.0
            newDist.normalize()    # 将信念分布归一化,实际上是得到概率相等的信念分布
            for newPosition, newProbability in newDist.items():   
                    updatedBeliefs[newPosition] += newProbability * oldProbability  # 与之前的信念分布相乘得到新的信念分布

        # 考虑保护的食物被吃掉
        lastState = self.getPreviousObservation()    # 上一次agent 移动时的游戏状态
        if lastState:
            lostFood = []
            # 对比前后状态的food位置的列表，找出丢失的食物
            for food in self.getFoodYouAreDefending(lastState).asList():
                if food not in self.getFoodYouAreDefending(gameState).asList(): 
                    lostFood.append(food)
            for f in lostFood:
                updatedBeliefs[f] = 1.0/len(self.getOpponents(gameState))   # 认为每个对手吃到的食物的概率是一样的

        self.positionBeliefs[agent] = updatedBeliefs    # 更新这个对手的信念分布


    def observe(self, agent, noisyDistance, gameState):
        """
        基于agent的噪声距离测量来更新代理位置的置信分布
        """
        myPosition = self.getAgentPosition(self.index, gameState)    # 获取我方位置
        teammatePositions = [self.getAgentPosition(teammate, gameState) 
                            for teammate in self.getTeam(gameState)]    # 获取我方agent位置列表
        updatedBeliefs = util.Counter() 

        for p in self.legalPositions:    # 遍历所有非墙壁位置
            if any([util.manhattanDistance(teammatePos, p) <= SIGHT_RANGE for teammatePos in teammatePositions]):
                    updatedBeliefs[p] = 0      # 曼哈顿距离小于5时,不需要用到噪声距离,概率置为0
            else:
                    trueDistance = util.manhattanDistance(myPosition, p)    # 真实距离
                    positionProbability = gameState.getDistanceProb(trueDistance, noisyDistance)    # 给定真实距离的噪声距离的概率
                    updatedBeliefs[p] = positionProbability * self.positionBeliefs[agent][p]    # 乘以对手的信念分布

        if not updatedBeliefs.totalCount():  # 如果信念分布全为0
            self.initializeBeliefs(agent)  # 初始化对手信念分布
        else:   
            updatedBeliefs.normalize()    # 归一化对手的信念分布 
            self.positionBeliefs[agent] = updatedBeliefs


    def guessPosition(self, agent):
        """
        返回游戏中给定代理最可能的位置
        """
        return self.positionBeliefs[agent].argMax()

    def expectinegamax(self, opponent, state, depth, sign, retAction=False):
        """
        opponent: 对手的索引
        state: 当前游戏状态
        depth: 搜索的深度
        sign: 用于在Negamax算法中切换agent的符号,初始值是1,先走我方agent
        retAction: 是否返回最佳动作
        """
        if sign == 1:
            agent = self.index
        else:
            agent = opponent

        bestAction = None
        if self.stateIsTerminal(agent, state) or depth == 0:    # 如果游戏结束或搜索深度为0
            bestVal = sign * self.evaluateState(state)    # 评估当前状态

        else:
            actions = state.getLegalActions(agent)    # 获取当前agent可以执行的动作
            actions.remove(Directions.STOP)    # 去掉停止动作
            bestVal = float("-inf") if agent == self.index else 0    # 己方初始值为负无穷,对手初始值为0
            for action in actions:    # 遍历所有动作
                    successor = state.generateSuccessor(agent, action)  # 得到动作后的状态
                    value = -self.expectinegamax(opponent, successor, depth - 1, -sign)      # 递归搜索
                    if agent == self.index and value > bestVal:     # 己方agent且当前动作值大于最佳值
                        bestVal, bestAction = value, action    # 更新最佳值和动作
                    elif agent == opponent:    # 对手agent
                        bestVal += value/len(actions)    # 对手agent,对所有动作求平均值

        if agent == self.index and retAction:    # 己方agent返回动作和值
            return bestVal, bestAction
        else:
            return bestVal  # 对手agent返回值

    def stateIsTerminal(self, agent, gameState):
        """
        agent是否有合法动作作为搜索树停止的条件
        """
        return len(gameState.getLegalActions(agent)) == 0

    def evaluateState(self, gameState):
        """
        评估游戏状态的效用
        """
        util.raiseNotDefined()

    # 通用的辅助函数
    def getAgentPosition(self, agent, gameState):
        """
        返回指定agent的位置
        """
        pos = gameState.getAgentPosition(agent)
        if pos:
            return pos
        else:
            return self.guessPosition(agent)    # 不确定就猜测

    def agentIsPacman(self, agent, gameState):
        """
        判断agent是否是pacman
        """
        agentPos = self.getAgentPosition(agent, gameState)
        return (gameState.isRed(agentPos) != gameState.isOnRedTeam(agent))


    def getOpponentDistances(self, gameState):
        """
        返回相对于此agent的对手的 ID 和距离
        """
        ID_and_distances = []
        my_position = self.getAgentPosition(self.index, gameState)
        for o in self.getOpponents(gameState):
            opponent_position = self.getAgentPosition(o, gameState)
            distance = self.distancer.getDistance(my_position, opponent_position)
            ID_and_distances.append((o, distance))
        return ID_and_distances





class DefensiveAgent(ApproximateAdversarialAgent):

    TERMINAL_STATE_VALUE = -1000000

    def stateIsTerminal(self, agent, gameState):
        return self.agentIsPacman(self.index, gameState)
  
    def evaluateState(self, gameState):
        #myPosition = self.getAgentPosition(self.index, gameState)
        if self.agentIsPacman(self.index, gameState):
                return DefensiveAgent.TERMINAL_STATE_VALUE

        score = 0
        pacmanState = [self.agentIsPacman(opponent, gameState) for opponent in self.getOpponents(gameState)]
        opponentDistances = self.getOpponentDistances(gameState)

        for isPacman, (id, distance) in zip(pacmanState, opponentDistances):
            if isPacman:
                    score -= 100000
                    score -= 5 * distance
            elif not any(pacmanState):
                    score -= distance

        return score
    
    # def evaluateState(self, gameState):
    #     if self.agentIsPacman(self.index, gameState):
    #         return DefensiveAgent.TERMINAL_STATE_VALUE

    #     myPosition = self.getAgentPosition(self.index, gameState)
    #     shieldedFood = self.getFoodYouAreDefending(gameState).asList()
    #     opponentPositions = [self.getAgentPosition(opponent, gameState)
    #                         for opponent in self.getOpponents(gameState)]

    #     if len(shieldedFood):
    #         opponentDistances = util.Counter()
    #         opponentTotalDistances = util.Counter()

    #         for f in shieldedFood:
    #             for o in opponentPositions:
    #                 distance = self.distancer.getDistance(f, o)
    #                 opponentDistances[(f, o)] = distance
    #                 opponentTotalDistances[o] -= distance

    #         threateningOpponent = opponentTotalDistances.argMax()
    #         atRiskFood, shortestDist = None, float("inf")
    #         for (food, opponent), dist in opponentDistances.items():
    #             if opponent == threateningOpponent and dist < shortestDist:
    #                 atRiskFood, shortestDist = food, dist

    #         return len(shieldedFood) \
    #                 - 2 * self.distancer.getDistance(myPosition, atRiskFood) \
    #                 - self.distancer.getDistance(myPosition, threateningOpponent)
    #     else:
    #         return -min(self.getOpponentDistances(gameState), key=lambda t: t[1])[1]




class OpportunisticAttackAgent(ApproximateAdversarialAgent):
    def evaluateState(self, gameState):
        myPosition = self.getAgentPosition(self.index, gameState)
        food = self.getFood(gameState).asList()

        targetFood = None
        maxDist = 0

        opponentDistances = self.getOpponentDistances(gameState)
        opponentDistance = min([dist for id, dist in opponentDistances])

        if not food or gameState.getAgentState(self.index).numCarrying > self.getScore(gameState) > 0:
            return 20 * self.getScore(gameState) \
                    - self.distancer.getDistance(myPosition, gameState.getInitialAgentPosition(self.index)) \
                    + opponentDistance

        for f in food:
            d = min([self.distancer.getDistance(self.getAgentPosition(o, gameState), f)
                    for o in self.getOpponents(gameState)])
            if d > maxDist:
                targetFood = f
                maxDist = d
        if targetFood:
            foodDist = self.distancer.getDistance(myPosition, targetFood)
        else:
            foodDist = 0

        distanceFromStart = abs(myPosition[0] - gameState.getInitialAgentPosition(self.index)[0])
        if not len(food):
            distanceFromStart *= -1
        return 2 * self.getScore(gameState)- 100 * len(food) - 2 * foodDist+ opponentDistance + distanceFromStart




class CautiousAttackAgent(ApproximateAdversarialAgent):
  """
  An attack-oriented agent that will retreat back to its home zone
  after consuming 5 pellets.
  """
  def registerInitialState(self, gameState):
        ApproximateAdversarialAgent.registerInitialState(self, gameState)
        self.retreating = False

  def chooseAction(self, gameState):
        if (gameState.getAgentState(self.index).numCarrying < 5 and
            len(self.getFood(gameState).asList())):
            self.retreating = False
        else:
            self.retreating = True

        return ApproximateAdversarialAgent.chooseAction(self, gameState)

  def evaluateState(self, gameState):
        myPosition = self.getAgentPosition(self.index, gameState)
        targetFood = self.getFood(gameState).asList()
        distanceFromStart = abs(myPosition[0] - gameState.getInitialAgentPosition(self.index)[0])
        opponentDistances = self.getOpponentDistances(gameState)
        opponentDistance = min([dist for id, dist in opponentDistances])

        if self.retreating:
            return  - len(targetFood) \
                    - 2 * distanceFromStart \
                    + opponentDistance
        else:
            foodDistances = [self.distancer.getDistance(myPosition, food)
                            for food in targetFood]
            minDistance = min(foodDistances) if len(foodDistances) else 0
        return 2 * self.getScore(gameState) \
                - 100 * len(targetFood) \
                - 3 * minDistance \
                + 2 * distanceFromStart \
                + opponentDistance












# class ReflexCaptureAgent(CaptureAgent):

#     def registerInitialState(self, gameState):
#         self.start = gameState.getAgentPosition(self.index)
#         self.foodCarried = 0
#         self.changeTopacman = []
#         self.depth = int(2)
#         CaptureAgent.registerInitialState(self, gameState)

#     def chooseAction(self, gameState):
#         actions = gameState.getLegalActions(self.index)
#         values = [self.evaluate(gameState, a) for a in actions]
#         maxValue = max(values)
#         bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        
#         """写决策树"""
#         foodLeft = len(self.getFood(gameState).asList())
#         self.recordChangeToPacman(gameState)

#         if foodLeft <= 2 or self.foodCarried >= 15:
#             bestDist = 9999
#             for action in actions:
#                 successor = self.getSuccessor(gameState, action)
#                 pos2 = successor.getAgentPosition(self.index)
#                 dist = self.getMazeDistance(self.changeTopacman[0],pos2)
#                 if dist < bestDist:
#                     bestAction = action
#                     bestDist = dist
#             return bestAction
        
#         return random.choice(bestActions)
    
#     def evaluate(self, gameState, action):
#         features = self.getFeatures(gameState, action)
#         weights = self.getWeights(gameState, action)
#         return features * weights

#     def recordChangeToPacman(self, gameState):
#         """
#         当上一个状态是在己方地区，而下一个状态是敌方地盘，记录上一个状态所在位置到列表self.changetopacman。
#         """
#         # 获取上一个状态
#         prevObservation = self.getPreviousObservation()
#         if prevObservation:
#             # 获取上一个状态时代理的位置
#             prevPos = prevObservation.getAgentState(self.index).getPosition()
#             # 检查代理在上一个状态和当前状态是否在敌方地盘
#             wasPacman = prevObservation.getAgentState(self.index).isPacman
#             isPacman = gameState.getAgentState(self.index).isPacman

#             # 当代理从己方地区进入敌方地盘时，记录上一个状态的位置
#             if not wasPacman and isPacman:
#                 self.changeTopacman.append(prevPos)
    
#     def getFoodCarried(self, gameState):
#         """
#         获取在敌方地盘吃到的食物数量，当agent被敌人吃掉或回到己方地盘时，计数清零。
#         """
#         # 获取当前代理的位置
#         myPos = gameState.getAgentState(self.index).getPosition()
#         # 获取上一个状态
#         prevObservation = self.getPreviousObservation()
#         if prevObservation:
#             # 获取上一个状态和当前状态的食物数量
#             prevFoodList = self.getFood(prevObservation).asList()
#             currFoodList = self.getFood(gameState).asList()

#             # 如果食物数量减少，增加食物数量
#             if len(prevFoodList) > len(currFoodList):
#                 self.foodCarried += 1

#             # 检查代理在上一个状态和当前状态是否在敌方地盘
#             wasPacman = prevObservation.getAgentState(self.index).isPacman
#             isPacman = gameState.getAgentState(self.index).isPacman

#             # 当代理被敌人吃掉或回到己方地盘时，计数清零
#             if (wasPacman and not isPacman) or (not wasPacman and not isPacman and self.getMazeDistance(myPos, self.start) == 0):
#                 self.foodCarried = 0

#         return self.foodCarried
    
#     def getSuccessor(self, gameState, action):
#         """
#         Finds the next successor which is a grid position (location tuple).
#         """
#         successor = gameState.generateSuccessor(self.index, action)
#         pos = successor.getAgentState(self.index).getPosition()
#         if pos != nearestPoint(pos):
#             # Only half a grid position was covered
#             return successor.generateSuccessor(self.index, action)
#         else:
#             return successor
        
# class offensiveReflexAgent(ReflexCaptureAgent):
#     def getFeatures(self, gameState, action):
#         features = util.Counter()
#         successor = self.getSuccessor(gameState, action)

#         myState = successor.getAgentState(self.index)
#         newPos = myState.getPosition()  # 新位置
#         newFood = self.getFood(successor).asList()     # 获取食物列表
#         newCapsules = self.getCapsules(successor)   # 获取胶囊列表
#         # 获取敌人状态列表
#         enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
#         newGhostStates = [a for a in enemies if (not a.isPacman) and a.getPosition() != None]

#         # 第一个feature：距离食物的距离
#         dist_to_foods = [self.getMazeDistance(newPos, food) for food in newFood]
#         nearestFood = min(dist_to_foods) if dist_to_foods else 0
#         features['foodVal'] = 1.0 / (nearestFood + 1)

#         # 第二个feature：距离敌人的距离
#         ghostVals = []
#         if len(newGhostStates) > 0:
#             for ghostState in newGhostStates:
#                 dist_to_ghost = self.getMazeDistance(newPos, ghostState.getPosition())
#                 if ghostState.scaredTimer > 5:
#                     ghostVal = (1.0 / (dist_to_ghost + 1)) + 2 * ghostState.scaredTimer
#                 else:
#                     ghostVal = -1.0 / (dist_to_ghost + 1)
#                 ghostVals.append(ghostVal)
#             features['ghostVal'] = sum(ghostVals)
#         else:
#             features['ghostVal'] = 0
            
#         # 第三个feature：距离胶囊的距离
#         dist_to_capsules = [self.getMazeDistance(newPos, capsule) for capsule in newCapsules]
#         nearestCapsule = min(dist_to_capsules) if dist_to_capsules else 0
#         features['capsuleVal'] = 1.0 / (nearestCapsule + 1)

#         # 第四个feature：身上的食物数量
#         features['FoodCarried'] = self.getFoodCarried(gameState)
#         return features
    
#     def getWeights(self, gameState, action):
#         return {'foodVal': 100, 'ghostVal': 10, 'capsuleVal': 10, 'FoodCarried' : 1}
    
    
    
# class MultiAgentSearchAgent(Agent):


#     def __init__(self, evalFn = 'betterEvaluationFunction', depth = '2'):
#         self.index = 0 # Pacman is always agent index 0
#         self.evaluationFunction = util.lookup(evalFn, globals())
#         self.depth = int(depth)

# class AlphaBetaAgent(ReflexCaptureAgent):
#     """
#     Your minimax agent with alpha-beta pruning (question 3)
#     """

#     def chooseAction(self, gameState):
#         """
#         Returns the minimax action using self.depth and self.evaluationFunction
#         """
#         # Initialize
#         alpha = float('-inf')
#         beta = float('inf')
#         _, best_action = self.max_value(gameState, self.index, alpha, beta)


#         return best_action

#     def max_value(self, gameState, depth, alpha, beta):
#         v = float('-inf')
#         best_action = Directions.STOP
#         for action in gameState.getLegalActions(self.index):
#             successor_value, _ = self.value(gameState.generateSuccessor(self.index, action), depth,alpha, beta)
#             if successor_value > v:
#                 v = successor_value
#                 best_action = action
#             # Due to pruning, early return
#             if v > beta:
#                 return v, best_action
#             alpha = max(alpha, v)
#         return v, best_action
    
#     def value(self, gameState, depth, alpha, beta):
#         return self.max_value(gameState, depth, alpha, beta)

        
#     def getSuccessor(self, gameState, action):
#         """
#         Finds the next successor which is a grid position (location tuple).
#         """
#         successor = gameState.generateSuccessor(self.index, action)
#         pos = successor.getAgentState(self.index).getPosition()
#         if pos != nearestPoint(pos):
#             # Only half a grid position was covered
#             return successor.generateSuccessor(self.index, action)
#         else:
#             return successor
    


# class ExpectimaxAgent(MultiAgentSearchAgent):
#     """
#       Your expectimax agent (question 4)
#     """

#     def chooseAction(self, gameState):

#         _, best_action = self.value(gameState, self.depth, self.index)
#         return best_action

#     def value(self, gameState, depth,):
#         return self.max_value(gameState, depth)


#     def max_value(self, gameState, depth):
#         v = float('-inf')
#         best_action = Directions.STOP
#         for action in gameState.getLegalActions(0):
#             successor_value, _ = self.value(gameState.generateSuccessor(0, action), depth, 1)
#             if successor_value > v:
#                 v = successor_value
#                 best_action = action
#         return v, best_action

#     def exp_value(self, gameState, depth, agentIndex):
#         v = 0
#         best_action = Directions.STOP
#         numAgents = gameState.getNumAgents()
#         legalActions = gameState.getLegalActions(agentIndex)
#         p = 1 / len(legalActions) # Probability
#         expectedMax_and_action = []

#         for action in legalActions:
#             # Check if the last ghost
#             if agentIndex == numAgents - 1:
#                 successor_value, _ = self.value(gameState.generateSuccessor(agentIndex, action), depth + 1, 0)
#             else:
#                 successor_value, action = self.value(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
#             # Append the value and action to the list
#             expectedMax_and_action.append((successor_value, action))
#             v += p * successor_value
#         # Find the max value of the ghosts', then use that action
#         best_action = max(expectedMax_and_action, key=lambda x: x[0])[1]
#         return v, best_action

# def betterEvaluationFunction(currentGameState):
#     """
#     Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
#     evaluation function (question 5).

#     DESCRIPTION: <write something here so we know what you did>
#     - distance to foods
#     - distance to ghosts
#     - distance to capsules
#     """
#     "*** YOUR CODE HERE ***"
#     newPos = currentGameState.getPacmanPosition()
#     newFood = currentGameState.getFood().asList()
#     newCapsules = currentGameState.getCapsules()
#     newGhostStates = currentGameState.getGhostStates()

#     # First feature: Position the the Foods
#     dist_to_foods = [util.manhattanDistance(newPos, food) for food in newFood]
#     nearestFood = min(dist_to_foods) if dist_to_foods else 0
#     foodVal = 1.0 / (nearestFood + 1)

#     # Second featue: Position to the ghosts
#     ghostVals = []
#     for ghostState in newGhostStates:
#         dist_to_ghost = util.manhattanDistance(newPos, ghostState.getPosition())
#         if ghostState.scaredTimer > 0:
#             # Encourage the pacman to eat ghosts by adding timer as value
#             ghostVal = (1.0 / (dist_to_ghost + 1)) + 2 * ghostState.scaredTimer
#         else:
#             ghostVal = -1.0 / (dist_to_ghost + 1)
#         ghostVals.append(ghostVal)

#     # Third feature: newPos to the capsules
#     dist_to_capsules = [util.manhattanDistance(newPos, capsule) for capsule in newCapsules]
#     nearestCapsule = min(dist_to_capsules) if dist_to_capsules else 0
#     capsuleVal = 1.0 / (nearestCapsule + 1)

#     return currentGameState.getScore() + 2 * foodVal + 2 * capsuleVal + 3 * sum(ghostVals)

# # 进攻型代理类
# class OffensiveAgent(CaptureAgent):
#     def registerInitialState(self, gameState):
#         # 记录初始位置
#         self.start = gameState.getAgentPosition(self.index)
#         # 调用父类方法进行初始化
#         CaptureAgent.registerInitialState(self, gameState)
#         # 初始化一个变量来记录吃掉的食物数量
#         self.foodCarried = 0
    
#     def chooseAction(self, gameState):
#         start = time.time()
#         # 获取所有合法动作
#         actions = gameState.getLegalActions(self.index)
#         # 评估每个动作的得分
#         values = [self.evaluate(gameState, a) for a in actions] 
#         # 找到最高得分
#         maxValue = max(values)
#         # 选择得分最高的动作
#         bestActions = [a for a, v in zip(actions, values) if v == maxValue]

#         # 获取剩余食物的数量
#         foodLeft = len(self.getFood(gameState).asList())

#         # 获取当前代理的位置
#         myPos = gameState.getAgentState(self.index).getPosition()

#         # 获取敌人的位置和状态
#         enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
#         # 入侵者
#         invaders= [a for a in enemies if  a.isPacman and a.getPosition() != None]
#         # 鬼
#         ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
#         scaredGhosts = [a for a in invaders if a.scaredTimer > 0]
#         activeGhosts = [a for a in invaders if a.scaredTimer == 0]

#         # 获取胶囊的位置
#         capsules = self.getCapsules(gameState)

#         # 如果pacman身上没有携带食物且场上存在安全的食物，进入食用安全的食物状态
#         if self.foodCarried == 0 and len(self.getSafeFood(gameState)) > 0:
#             bestDist = 9999
#             safeFood = self.getSafeFood(gameState)
#             for action in actions:
#                 successor = self.getSuccessor(gameState, action)
#                 pos2 = successor.getAgentPosition(self.index)
#                 dist = min([self.getMazeDistance(pos2, food) for food in safeFood])
#                 if dist < bestDist:
#                     bestAction = action
#                     bestDist = dist
#             #print ('eval time for agent %d: %.4f' % (self.index, time.time() - start))
#             return bestAction
#         Changepos = self.getpreLocation_GhostToPacman(gameState)
#         # 如果吃掉的食物数量大于等于15或剩余食物少于等于2个或游戏时间只剩下回家的时间，优先返回起始位置
#         if self.foodCarried >= 15 or foodLeft <= 2 or gameState.data.timeleft < self.getMazeDistance(myPos, self.start) + 100:
#             bestDist = 9999
#             for action in actions:
#                 successor = self.getSuccessor(gameState, action)
#                 pos2 = successor.getAgentPosition(self.index)
#                 dist = self.getMazeDistance(self.start, pos2)
#                 if dist < bestDist:
#                     bestAction = action
#                     bestDist = dist
#             # 如果已经返回到起始位置，重置吃掉的食物数量
#             if bestDist == 0:
#                 self.foodCarried = 0
#             #print ('eval time for agent %d: %.4f' % (self.index, time.time() - start))
#             return bestAction

#         # 如果场上没有安全的食物，场上还有胶囊，对手处于恐吓剩下的时间不多时，进入搜索胶囊状态
#         if len(self.getSafeFood(gameState)) == 0 and len(capsules) > 0 and any([ghost.scaredTimer < 10 for ghost in scaredGhosts]):
#             bestDist = 9999
#             for action in actions:
#                 successor = self.getSuccessor(gameState, action)
#                 pos2 = successor.getAgentPosition(self.index)
#                 dist = min([self.getMazeDistance(pos2, capsule) for capsule in capsules])
#                 if dist < bestDist:
#                     bestAction = action
#                     bestDist = dist
#             #print ('eval time for agent %d: %.4f' % (self.index, time.time() - start))
#             return bestAction



#         # 如果pacman身上没有携带食物且场上没有安全的食物，直接去搜索食物
#         if self.foodCarried == 0 and len(self.getSafeFood(gameState)) == 0:
#             bestDist = 9999
#             for action in actions:
#                 successor = self.getSuccessor(gameState, action)
#                 pos2 = successor.getAgentPosition(self.index)
#                 dist = min([self.getMazeDistance(pos2, food) for food in self.getFood(gameState).asList()])
#                 if dist < bestDist:
#                     bestAction = action
#                     bestDist = dist
#             #print ('eval time for agent %d: %.4f' % (self.index, time.time() - start))
#             return bestAction

#         # 如果鬼出现且与pacman相距不远，进入逃生状态
#         if len(activeGhosts) > 0 and min([self.getMazeDistance(myPos, ghost.getPosition()) for ghost in activeGhosts]) < 5:
#             bestDist = -9999
#             for action in actions:
#                 successor = self.getSuccessor(gameState, action)
#                 pos2 = successor.getAgentPosition(self.index)
#                 dist = min([self.getMazeDistance(pos2, ghost.getPosition()) for ghost in activeGhosts])
#                 if dist > bestDist:
#                     bestAction = action
#                     bestDist = dist
#             #print ('eval time for agent %d: %.4f' % (self.index, time.time() - start))
#             return bestAction

#         # 如果大力丸的持续时间足够，进入搜索危险食物状态
#         if len(scaredGhosts) > 0 and max([ghost.scaredTimer for ghost in scaredGhosts]) > 20:
#             bestDist = 9999
#             for action in actions:
#                 successor = self.getSuccessor(gameState, action)
#                 pos2 = successor.getAgentPosition(self.index)
#                 dist = min([self.getMazeDistance(pos2, food) for food in self.getFood(gameState).asList()])
#                 if dist < bestDist:
#                     bestAction = action
#                     bestDist = dist
#             return bestAction

#         # 否则，从得分最高的动作中随机选择一个
#         return random.choice(bestActions)
    
#     def getpreLocation_GhostToPacman(self,gameState):
#         """当agent从ghost变为pacman时，获取上一个状态的位置"""
#         prevObservation = self.getPreviousObservation()
#         if prevObservation:
#             # 检查代理在上一个状态和当前状态是否是Pacman
#             wasPacman = prevObservation.getAgentState(self.index).isPacman
#             isPacman = gameState.getAgentState(self.index).isPacman

#             # 当上一个状态不是Pacman，而当前状态是Pacman时，获取上一个状态的位置
#             if not wasPacman and isPacman:
#                 # 获取上一个状态时代理的位置
#                 prevPos = prevObservation.getAgentState(self.index).getPosition()
#                 return prevPos
#         return None
#     def getSuccessor(self, gameState, action):
#         # 获取当前动作的后继状态
#         successor = gameState.generateSuccessor(self.index, action)
#         pos = successor.getAgentState(self.index).getPosition()
#         # 如果位置不是整数点，再次生成后继状态
#         if pos != nearestPoint(pos):
#             return successor.generateSuccessor(self.index, action)
#         else:
#             return successor

#     def evaluate(self, gameState, action):
#         # 计算特征值和特征权重的线性组合
#         features = self.getFeatures(gameState, action)
#         weights = self.getWeights(gameState, action)
#         return features * weights

#     def getFeatures(self, gameState, action):
#         # 获取当前动作的特征
#         features = util.Counter()
#         successor = self.getSuccessor(gameState, action)
#         foodList = self.getFood(successor).asList()    
#         features['successorScore'] = -len(foodList)

#         # 计算到最近食物的距离
#         if len(foodList) > 0:
#             myPos = successor.getAgentState(self.index).getPosition()
#             minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
#             features['distanceToFood'] = minDistance

#         # 计算到敌方入侵者的距离
#         enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
#         invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
#         if len(invaders) > 0:
#             dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
#             features['invaderDistance'] = min(dists)

#         # 如果吃掉了食物，增加吃掉的食物数量
#         if action in gameState.getLegalActions(self.index):
#             nextState = gameState.generateSuccessor(self.index, action)
#             nextFoodList = self.getFood(nextState).asList()
#             if len(nextFoodList) < len(foodList):
#                 self.foodCarried += 1

#         return features

#     def getWeights(self, gameState, action):
#         # 定义特征的权重
#         return {'successorScore': 100, 'distanceToFood': -1, 'invaderDistance': -10}

#     def getSafeFood(self, gameState):
#         # 获取安全的食物，即没有鬼靠近的食物
#         foodList = self.getFood(gameState).asList()
#         myPos = gameState.getAgentState(self.index).getPosition()
#         enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
#         ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
#         safeFood = []
#         # 遍历所有食物，检查是否有鬼靠近
#         for food in foodList:
#             safe = True
#             for ghost in ghosts:
#                 if self.getMazeDistance(food, ghost.getPosition()) < 5:
#                     safe = False
#                     break
#             if safe:
#                 safeFood.append(food)
#         return safeFood


# # 防守型代理类
# class DefensiveAgent(CaptureAgent):
#     def registerInitialState(self, gameState):
#         # 记录初始位置
#         self.start = gameState.getAgentPosition(self.index)
#         # 调用父类方法进行初始化
#         CaptureAgent.registerInitialState(self, gameState)
    
#     def chooseAction(self, gameState):
#         # 获取所有合法动作
#         actions = gameState.getLegalActions(self.index)
#         # 评估每个动作的得分
#         values = [self.evaluate(gameState, a) for a in actions]
#         # 找到最高得分
#         maxValue = max(values)
#         # 选择得分最高的动作
#         bestActions = [a for a, v in zip(actions, values) if v == maxValue]

#         # 从得分最高的动作中随机选择一个
#         return random.choice(bestActions)

#     def getSuccessor(self, gameState, action):
#         # 获取当前动作的后继状态
#         successor = gameState.generateSuccessor(self.index, action)
#         pos = successor.getAgentState(self.index).getPosition()
#         # 如果位置不是整数点，再次生成后继状态
#         if pos != nearestPoint(pos):
#             return successor.generateSuccessor(self.index, action)
#         else:
#             return successor

#     def evaluate(self, gameState, action):
#         # 计算特征值和特征权重的线性组合
#         features = self.getFeatures(gameState, action)
#         weights = self.getWeights(gameState, action)
#         return features * weights

#     def getFeatures(self, gameState, action):
#         # 获取当前动作的特征
#         features = util.Counter()
#         successor = self.getSuccessor(gameState, action)

#         myState = successor.getAgentState(self.index)
#         myPos = myState.getPosition()

#         # 计算是否在防守状态
#         features['onDefense'] = 1
#         if myState.isPacman: features['onDefense'] = 0

#         # 计算到敌方入侵者的距离
#         enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
#         invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
#         features['numInvaders'] = len(invaders)
#         if len(invaders) > 0:
#             dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
#             features['invaderDistance'] = min(dists)

#         # 如果动作是停止或者反方向，增加惩罚
#         if action == Directions.STOP: features['stop'] = 1
#         rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
#         if action == rev: features['reverse'] = 1

#         return features

#     def getWeights(self, gameState, action):
#         # 定义特征的权重
#         return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}



