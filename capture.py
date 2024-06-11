# capture.py
# ----------
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


# capture.py
# ----------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# ------------------------------------ 描述 ------------------------------------ #
#这是本地运行游戏的主文件。
# 文件中的 GameState 类提供了许多获得当前游戏的状态信息（包括食物点、能量胶囊、智能体配置信息等）的函数。
#该文件还描述了游戏的运行逻辑。
#可以通过这个类来获取游戏运行信息
# ---------------------------------------------------------------------------- #



"""
Capture.py holds the logic for Pacman capture the flag.

    (i)  Your interface to the pacman world:
                    Pacman is a complex environment.  You probably don't want to
                    read through all of the code we wrote to make the game runs
                    correctly.  This section contains the parts of the code
                    that you will need to understand in order to complete the
                    project.  There is also some code in game.py that you should
                    understand.

    (ii)  The hidden secrets of pacman:
                    This section contains all of the logic code that the pacman
                    environment uses to decide who can move where, who dies when
                    things collide, etc.  You shouldn't need to read this section
                    of code, but you can if you want.

    (iii) Framework to start a game:
                    The final section contains the code for reading the command
                    you use to set up the game, then starting up a new game, along with
                    linking in all the external parts (agent functions, graphics).
                    Check this section out to see all the options available to you.

To play your first game, type 'python capture.py' from the command line.
The keys are
    P1: 'a', 's', 'd', and 'w' to move
    P2: 'l', ';', ',' and 'p' to move
    
翻译：
Capture.py 持有 Pacman 夺旗的逻辑。

（i） 您与吃豆子世界的接口：
                    吃豆人是一个复杂的环境。 你可能不想
                    通读我们编写的所有代码，以使游戏运行
                    正确。 本部分包含代码的各个部分
                    您需要了解才能完成
                    项目。 game.py 中还有一些代码，你应该这样做
                    理解。

（ii） 吃豆人隐藏的秘密：
                    本节包含 pacman 的所有逻辑代码
                    环境用来决定谁可以搬到哪里，谁什么时候死
                    事物碰撞等。 您不需要阅读此部分
                    的代码，但如果你愿意，你可以。

（iii） 启动游戏的框架：
                    最后一部分包含用于读取命令的代码
                    你用来设置游戏，然后启动一个新游戏，以及
                    链接所有外部部件（代理功能、图形）。
                    查看此部分以查看所有可用的选项。

要玩您的第一个游戏，请从命令行键入“python capture.py”。
关键是
    P1：'a'、's'、'd' 和 'w' 移动
    P2： 'l'， ';'， '，' 和 'p' 移动
"""

from game import GameStateData
from game import Game
from game import Directions
from game import Actions
from util import nearestPoint
from util import manhattanDistance
from game import Grid
from game import Configuration
from game import Agent
from game import reconstituteGrid
from mazeGenerator import generateMaze
import sys, util, types, time, random, os
import importlib
import keyboardAgents   
import layout
from glob import glob
# If you change these, you won't affect the server, so you can't cheat
KILL_POINTS = 0
SONAR_NOISE_RANGE = 13 # Must be odd
SONAR_NOISE_VALUES = [i - (SONAR_NOISE_RANGE - 1)//2 for i in range(SONAR_NOISE_RANGE)]
SIGHT_RANGE = 5 # Manhattan distance
MIN_FOOD = 2
TOTAL_FOOD = 60

DUMP_FOOD_ON_DEATH = True # if we have the gameplay element that dumps dots on death

SCARED_TIME = 40

def noisyDistance(pos1, pos2):
    return int(util.manhattanDistance(pos1, pos2) + random.choice(SONAR_NOISE_VALUES))

###################################################
# YOUR INTERFACE TO THE PACMAN WORLD: A GameState #
###################################################

# ---------------------------------------------------------------------------- #
#游戏状态类，指定完整的游戏状态，包括食物、胶囊、agent配置和分数更改。
# GameStates 由 Game 对象用来捕捉游戏的实际状态，并可供agent用来推理游戏。
#包含的方法：
#getLegalActions，返回agent特定的合法动作
#generateSuccessor，返回指定agent采取行动后的后继状态（GameState 对象）。
#getAgentState，返回agent状态
#getAgentPosition，如果具有给定索引的agent是可观察的，则返回位置元组；如果agent不可观察，则返回 None。
#getNumAgents，返回agent的数量
#getScore，返回与当前分数相对应的数字。
#getRedFood，返回与红队一方的食物相对应的食物矩阵。对于矩阵 m，如果 (x,y) 中有属于红队的食物（即红队正在保护它，蓝队正在试图吃掉它），则 m[x][y]=true。
#getBlueFood，返回与蓝队一方的食物相对应的食物矩阵。对于矩阵 m，如果 (x,y) 中有属于蓝队的食物（即蓝队正在保护它，红队正在试图吃掉它），则 m[x][y]=true。
#getRedCapsules，返回红色胶囊对应的胶囊矩阵
#getBlueCapsules，返回蓝色胶囊对应的胶囊矩阵
#getWalls，获得墙壁矩阵
#hasFood，如果位置 (x,y) 有食物，则返回 true，无论它是蓝队食物还是红队食物。
#hasWall，如果位置 (x,y) 有墙壁，则返回 true
#isOver，返回游戏是否结束？
#getRedTeamIndices，返回红队agent索引列表。
#getBlueTeamIndices，返回蓝队agent索引列表。
#isOnRedTeam，如果具有给定 agentIndex 的agent在红队，则返回 true。
#getAgentDistances，返回每个agent的噪声距离。
#getDistanceProb，返回给定真实距离的噪声距离的概率
#getInitialAgentPosition，获取初始代理位置
#getCapsules，返回剩余胶囊的位置 (x,y) 列表。

# 关键是了解这些方法返回的东西具体是什么? 现在还不清楚
# ---------------------------------------------------------------------------- #
class GameState:
    """
    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes.
    GameState指定了完整的游戏状态，包括食物、药丸、agent配置和分数变化。

    GameStates are used by the Game object to capture the actual state of the game and
    can be used by agents to reason about the game.
    游戏状态用于由Game对象捕获游戏的实际状态，并可以被agent用来推理游戏。

    Much of the information in a GameState is stored in a GameStateData object.  We
    strongly suggest that you access that data via the accessor methods below rather
    than referring to the GameStateData object directly.
    GameState中的许多信息存储在GameStateData对象中。我们强烈建议您通过下面的访问器方法访问这些数据，而不是直接引用GameStateData对象。
    """

    ####################################################
    # Accessor methods: use these to access state data #
    ####################################################

    def getLegalActions( self, agentIndex=0 ):
        """
        Returns the legal actions for the agent specified.
        """
        return AgentRules.getLegalActions( self, agentIndex )

    def generateSuccessor( self, agentIndex, action):
        """
        Returns the successor state (a GameState object) after the specified agent takes the action.
        返回指定agent采取行动后的后继状态（GameState 对象）。
        """
        # Copy current state
        state = GameState(self)

        # Find appropriate rules for the agent
        AgentRules.applyAction( state, action, agentIndex )
        AgentRules.checkDeath(state, agentIndex)
        AgentRules.decrementTimer(state.data.agentStates[agentIndex])

        # Book keeping
        state.data._agentMoved = agentIndex
        state.data.score += state.data.scoreChange
        state.data.timeleft = self.data.timeleft - 1
        return state

    def getAgentState(self, index):
        return self.data.agentStates[index]

    def getAgentPosition(self, index):
        """
        Returns a location tuple if the agent with the given index is observable;
        if the agent is unobservable, returns None.
        如果具有给定索引的agent是可观察的，则返回位置元组；如果代理不可观察，则返回 None。
        """
        agentState = self.data.agentStates[index]
        ret = agentState.getPosition()
        if ret:
            return tuple(int(x) for x in ret)
        return ret

    def getNumAgents( self ):
        return len( self.data.agentStates )

    def getScore( self ):
        """
        Returns a number corresponding to the current score.
        返回与当前分数对应的数字。
        """
        return self.data.score

    def getRedFood(self):
        """
        Returns a matrix of food that corresponds to the food on the red team's side.
        For the matrix m, m[x][y]=true if there is food in (x,y) that belongs to
        red (meaning red is protecting it, blue is trying to eat it).
        返回与红队一方的食物相对应的食物矩阵。
        对于矩阵 m，如果 (x,y) 中有属于红队的食物（即红队正在保护它，蓝队正在试图吃掉它），则 m[x][y]=true
        """
        return halfGrid(self.data.food, red = True)

    def getBlueFood(self):
        """
        Returns a matrix of food that corresponds to the food on the blue team's side.
        For the matrix m, m[x][y]=true if there is food in (x,y) that belongs to
        blue (meaning blue is protecting it, red is trying to eat it).
        返回与蓝队一方的食物相对应的食物矩阵。
        对于矩阵 m，如果 (x,y) 中有属于蓝队的食物（即蓝队正在保护它，红队正在试图吃掉它），则 m[x][y]=true。
        """
        return halfGrid(self.data.food, red = False)

    def getRedCapsules(self):
        """获得红色胶囊"""
        return halfList(self.data.capsules, self.data.food, red = True)

    def getBlueCapsules(self):
        return halfList(self.data.capsules, self.data.food, red = False)

    def getWalls(self):
        """
        Just like getFood but for walls
        类似于 getFood，但用于墙壁
        """
        return self.data.layout.walls

    def hasFood(self, x, y):
        """
        Returns true if the location (x,y) has food, regardless of
        whether it's blue team food or red team food.
        如果位置 (x,y) 有食物，则返回 true，无论是蓝队食物还是红队食物。
        """
        return self.data.food[x][y]

    def hasWall(self, x, y):
        """
        Returns true if (x,y) has a wall, false otherwise.
        如果 (x,y) 有墙则返回 true，否则返回 false。
        """
        return self.data.layout.walls[x][y]

    def isOver( self ):
        return self.data._win

    def getRedTeamIndices(self):
        """
        Returns a list of agent index numbers for the agents on the red team.
        返回红队agent的索引号列表。
        """
        return self.redTeam[:]

    def getBlueTeamIndices(self):
        """
        Returns a list of the agent index numbers for the agents on the blue team.
        返回蓝队agent的索引号列表。
        """
        return self.blueTeam[:]

    def isOnRedTeam(self, agentIndex):
        """
        Returns true if the agent with the given agentIndex is on the red team.
        如果具有指定 agentIndex 的agent在红队，则返回 true
        """
        return self.teams[agentIndex]

    def getAgentDistances(self):
        """
        Returns a noisy distance to each agent.
        返回每个agent的噪声距离。   ???什么是噪声距离
        """
        if 'agentDistances' in dir(self) :
            return self.agentDistances
        else:
            return None

    def getDistanceProb(self, trueDistance, noisyDistance):
        "Returns the probability of a noisy distance given the true distance"
        #返回给定真实距离的噪声距离的概率
        if noisyDistance - trueDistance in SONAR_NOISE_VALUES:
            return 1.0/SONAR_NOISE_RANGE
        else:
            return 0

    def getInitialAgentPosition(self, agentIndex):
        "Returns the initial position of an agent."
        #返回代理的初始位置。
        return self.data.layout.agentPositions[agentIndex][1]

    def getCapsules(self):
        """
        Returns a list of positions (x,y) of the remaining capsules.
        返回剩余胶囊的位置 (x,y) 列表。
        """
        return self.data.capsules

    #############################################
    #             Helper methods:               #
    # You shouldn't need to call these directly #
    #       你不需要直接调用这些方法              #
    #############################################

    def __init__( self, prevState = None ):
        """
        Generates a new state by copying information from its predecessor.
        通过复制前身的信息来生成新状态
        """
        if prevState != None: # Initial state
            self.data = GameStateData(prevState.data)
            self.blueTeam = prevState.blueTeam
            self.redTeam = prevState.redTeam
            self.data.timeleft = prevState.data.timeleft

            self.teams = prevState.teams
            self.agentDistances = prevState.agentDistances
        else:
            self.data = GameStateData()
            self.agentDistances = []

    def deepCopy( self ):
        state = GameState( self )
        state.data = self.data.deepCopy()
        state.data.timeleft = self.data.timeleft

        state.blueTeam = self.blueTeam[:]
        state.redTeam = self.redTeam[:]
        state.teams = self.teams[:]
        state.agentDistances = self.agentDistances[:]
        return state

    def makeObservation(self, index):
        state = self.deepCopy()

        # Adds the sonar signal 添加声纳信号
        pos = state.getAgentPosition(index)
        n = state.getNumAgents()
        distances = [noisyDistance(pos, state.getAgentPosition(i)) for i in range(n)]
        state.agentDistances = distances

        # Remove states of distant opponents    删除远方对手的状态
        if index in self.blueTeam:
            team = self.blueTeam
            otherTeam = self.redTeam
        else:
            otherTeam = self.blueTeam
            team = self.redTeam

        for enemy in otherTeam:
            seen = False
            enemyPos = state.getAgentPosition(enemy)
            for teammate in team:
                if util.manhattanDistance(enemyPos, state.getAgentPosition(teammate)) <= SIGHT_RANGE:
                    seen = True
            if not seen: state.data.agentStates[enemy].configuration = None
        return state

    def __eq__( self, other ):
        """
        Allows two states to be compared.允许比较两个状态
        """
        if other == None: return False
        return self.data == other.data

    def __hash__( self ):
        """
        Allows states to be keys of dictionaries.允许状态成为字典的键。
        """
        return int(hash( self.data ))

    def __str__( self ):

        return str(self.data)

    def initialize( self, layout, numAgents):
        """
        Creates an initial game state from a layout array (see layout.py).
        从布局数组创建初始游戏状态（参见layout.py）。
        """
        self.data.initialize(layout, numAgents)
        positions = [a.configuration for a in self.data.agentStates]
        self.blueTeam = [i for i,p in enumerate(positions) if not self.isRed(p)]
        self.redTeam = [i for i,p in enumerate(positions) if self.isRed(p)]
        self.teams = [self.isRed(p) for p in positions]
        #This is usually 60 (always 60 with random maps)
        #However, if layout map is specified otherwise, it could be less
        global TOTAL_FOOD
        TOTAL_FOOD = layout.totalFood

    def isRed(self, configOrPos):
        width = self.data.layout.width
        if type(configOrPos) == type( (0,0) ):
            return configOrPos[0] < width / 2
        else:
            return configOrPos.pos[0] < width / 2


### 非GameSatate类的方法
def halfGrid(grid, red):    
    """如果red参数为True，则返回原始地图网格的左半部分；如果为False，则返回右半部分。"""
    halfway = grid.width // 2
    halfgrid = Grid(grid.width, grid.height, False)
    if red:    xrange = list(range(halfway))
    else:       xrange = list(range(halfway, grid.width))

    for y in range(grid.height):
        for x in xrange:
            if grid[x][y]: halfgrid[x][y] = True

    return halfgrid

def halfList(l, grid, red):
    """
    根据传入的布尔值red、网格grid的宽度的一半，以及列表l中的元组，
    创建一个新的列表newList，其中包含(x,y)的元组
    """
    halfway = grid.width / 2
    newList = []
    for x,y in l:
        if red and x <= halfway: newList.append((x,y))
        elif not red and x > halfway: newList.append((x,y))
    return newList

############################################################################
#                     THE HIDDEN SECRETS OF PACMAN                         #
#                           吃豆人隐藏的秘密                                #
# You shouldn't need to look through the code in this section of the file. #
#                          你不需要查看本文件中这一部分代码。                  #
############################################################################

COLLISION_TOLERANCE = 0.7 # How close ghosts must be to Pacman to kill

class CaptureRules:
    """
    These game rules manage the control flow of a game, deciding when
    and how the game starts and ends.
    """

    def __init__(self, quiet = False):
        self.quiet = quiet

    def newGame( self, layout, agents, display, length, muteAgents, catchExceptions ):
        initState = GameState()
        initState.initialize( layout, len(agents) )
        starter = random.randint(0,1)
        print(('%s team starts' % ['Red', 'Blue'][starter]))
        game = Game(agents, display, self, startingIndex=starter, muteAgents=muteAgents, catchExceptions=catchExceptions)
        game.state = initState
        game.length = length
        game.state.data.timeleft = length
        if 'drawCenterLine' in dir(display):
            display.drawCenterLine()
        self._initBlueFood = initState.getBlueFood().count()
        self._initRedFood = initState.getRedFood().count()
        return game

    def process(self, state, game):
        """
        Checks to see whether it is time to end the game.
        """
        if 'moveHistory' in dir(game):
            if len(game.moveHistory) == game.length:
                state.data._win = True

        if state.isOver():
            game.gameOver = True
            if not game.rules.quiet:
                redCount = 0
                blueCount = 0
                foodToWin = (TOTAL_FOOD//2) - MIN_FOOD
                for index in range(state.getNumAgents()):
                    agentState = state.data.agentStates[index]
                    if index in state.getRedTeamIndices():
                        redCount += agentState.numReturned
                    else:
                        blueCount += agentState.numReturned

                if blueCount >= foodToWin:#state.getRedFood().count() == MIN_FOOD:
                    print('The Blue team has returned at least %d of the opponents\' dots.' % foodToWin)
                elif redCount >= foodToWin:#state.getBlueFood().count() == MIN_FOOD:
                    print('The Red team has returned at least %d of the opponents\' dots.' % foodToWin)
                else:#if state.getBlueFood().count() > MIN_FOOD and state.getRedFood().count() > MIN_FOOD:
                    print('Time is up.')
                    if state.data.score == 0: print('Tie game!')
                    else:
                        winner = 'Red'
                        if state.data.score < 0: winner = 'Blue'
                        print('The %s team wins by %d points.' % (winner, abs(state.data.score)))

    def getProgress(self, game):
        blue = 1.0 - (game.state.getBlueFood().count() / float(self._initBlueFood))
        red = 1.0 - (game.state.getRedFood().count() / float(self._initRedFood))
        moves = len(self.moveHistory) / float(game.length)

        # return the most likely progress indicator, clamped to [0, 1]
        return min(max(0.75 * max(red, blue) + 0.25 * moves, 0.0), 1.0)

    def agentCrash(self, game, agentIndex):
        if agentIndex % 2 == 0:
            print("Red agent crashed", file=sys.stderr)
            game.state.data.score = -1
        else:
            print("Blue agent crashed", file=sys.stderr)
            game.state.data.score = 1

    def getMaxTotalTime(self, agentIndex):
        return 900  # Move limits should prevent this from ever happening

    def getMaxStartupTime(self, agentIndex):
        return 15 # 15 seconds for registerInitialState

    def getMoveWarningTime(self, agentIndex):
        return 1  # One second per move

    def getMoveTimeout(self, agentIndex):
        return 3  # Three seconds results in instant forfeit

    def getMaxTimeWarnings(self, agentIndex):
        return 2  # Third violation loses the game

class AgentRules:
    """
    These functions govern how each agent interacts with her environment.
    这些函数控制每个agent如何与其环境交互
    """

    def getLegalActions( state, agentIndex ):
        """
        Returns a list of legal actions (which are both possible & allowed)
        """
        agentState = state.getAgentState(agentIndex)
        conf = agentState.configuration
        possibleActions = Actions.getPossibleActions( conf, state.data.layout.walls )
        return AgentRules.filterForAllowedActions( agentState, possibleActions)
    getLegalActions = staticmethod( getLegalActions )

    def filterForAllowedActions(agentState, possibleActions):
        return possibleActions
    filterForAllowedActions = staticmethod( filterForAllowedActions )


    def applyAction( state, action, agentIndex ):
        """
        Edits the state to reflect the results of the action.
        """
        legal = AgentRules.getLegalActions( state, agentIndex )
        if action not in legal:
            raise Exception("Illegal action " + str(action))

        # Update Configuration
        agentState = state.data.agentStates[agentIndex]
        speed = 1.0
        # if agentState.isPacman: speed = 0.5
        vector = Actions.directionToVector( action, speed )
        oldConfig = agentState.configuration
        agentState.configuration = oldConfig.generateSuccessor( vector )

        # Eat
        next = agentState.configuration.getPosition()
        nearest = nearestPoint( next )

        if next == nearest:
            isRed = state.isOnRedTeam(agentIndex)
            # Change agent type
            agentState.isPacman = [isRed, state.isRed(agentState.configuration)].count(True) == 1
            # if he's no longer pacman, he's on his own side, so reset the num carrying timer
            #agentState.numCarrying *= int(agentState.isPacman)
            if agentState.numCarrying > 0 and not agentState.isPacman:
                score = agentState.numCarrying if isRed else -1*agentState.numCarrying
                state.data.scoreChange += score

                agentState.numReturned += agentState.numCarrying
                agentState.numCarrying = 0

                redCount = 0
                blueCount = 0
                for index in range(state.getNumAgents()):
                    agentState = state.data.agentStates[index]
                    if index in state.getRedTeamIndices():
                        redCount += agentState.numReturned
                    else:
                        blueCount += agentState.numReturned
                if redCount >= (TOTAL_FOOD//2) - MIN_FOOD or blueCount >= (TOTAL_FOOD//2) - MIN_FOOD:
                    state.data._win = True


        if agentState.isPacman and manhattanDistance( nearest, next ) <= 0.9 :
            AgentRules.consume( nearest, state, state.isOnRedTeam(agentIndex) )

    applyAction = staticmethod( applyAction )

    def consume( position, state, isRed ):
        x,y = position
        # Eat food
        if state.data.food[x][y]:

            # blue case is the default
            teamIndicesFunc = state.getBlueTeamIndices
            score = -1
            if isRed:
                # switch if its red
                score = 1
                teamIndicesFunc = state.getRedTeamIndices

            # go increase the variable for the pacman who ate this
            agents = [state.data.agentStates[agentIndex] for agentIndex in teamIndicesFunc()]
            for agent in agents:
                if agent.getPosition() == position:
                    agent.numCarrying += 1
                    break # the above should only be true for one agent...

            # do all the score and food grid maintainenace
            #state.data.scoreChange += score
            state.data.food = state.data.food.copy()
            state.data.food[x][y] = False
            state.data._foodEaten = position
            #if (isRed and state.getBlueFood().count() == MIN_FOOD) or (not isRed and state.getRedFood().count() == MIN_FOOD):
            #  state.data._win = True

        # Eat capsule
        if isRed: myCapsules = state.getBlueCapsules()
        else: myCapsules = state.getRedCapsules()
        if( position in myCapsules ):
            state.data.capsules.remove( position )
            state.data._capsuleEaten = position

            # Reset all ghosts' scared timers
            if isRed: otherTeam = state.getBlueTeamIndices()
            else: otherTeam = state.getRedTeamIndices()
            for index in otherTeam:
                state.data.agentStates[index].scaredTimer = SCARED_TIME

    consume = staticmethod( consume )

    def decrementTimer(state):
        timer = state.scaredTimer
        if timer == 1:
            state.configuration.pos = nearestPoint( state.configuration.pos )
        state.scaredTimer = max( 0, timer - 1 )
    decrementTimer = staticmethod( decrementTimer )

    def dumpFoodFromDeath(state, agentState, agentIndex):
        if not (DUMP_FOOD_ON_DEATH):
            # this feature is not turned on
            return

        if not agentState.isPacman:
            raise Exception('something is seriously wrong, this agent isnt a pacman!')

        # ok so agentState is this:
        if (agentState.numCarrying == 0):
            return

        # first, score changes!
        # we HACK pack that ugly bug by just determining if its red based on the first position
        # to die...
        dummyConfig = Configuration(agentState.getPosition(), 'North')
        isRed = state.isRed(dummyConfig)

        # the score increases if red eats dots, so if we are refunding points,
        # the direction should be -1 if the red agent died, which means he dies
        # on the blue side
        scoreDirection = (-1)**(int(isRed) + 1)
        #state.data.scoreChange += scoreDirection * agentState.numCarrying

        def onRightSide(state, x, y):
            dummyConfig = Configuration((x, y), 'North')
            return state.isRed(dummyConfig) == isRed

        # we have food to dump
        # -- expand out in BFS. Check:
        #   - that it's within the limits
        #   - that it's not a wall
        #   - that no other agents are there
        #   - that no power pellets are there
        #   - that it's on the right side of the grid
        def allGood(state, x, y):
            width, height = state.data.layout.width, state.data.layout.height
            food, walls = state.data.food, state.data.layout.walls

            # bounds check
            if x >= width or y >= height or x <= 0 or y <= 0:
                return False

            if walls[x][y]:
                return False
            if food[x][y]:
                return False

            # dots need to be on the side where this agent will be a pacman :P
            if not onRightSide(state, x, y):
                return False

            if (x,y) in state.data.capsules:
                return False

            # loop through agents
            agentPoses = [state.getAgentPosition(i) for i in range(state.getNumAgents())]
            if (x,y) in agentPoses:
                return False

            return True

        numToDump = agentState.numCarrying
        state.data.food = state.data.food.copy()
        foodAdded = []

        def genSuccessors(x, y):
            DX = [-1, 0, 1]
            DY = [-1, 0, 1]
            return [(x + dx, y + dy) for dx in DX for dy in DY]

        # BFS graph search
        positionQueue = [agentState.getPosition()]
        seen = set()
        while numToDump > 0:
            if not len(positionQueue):
                raise Exception('Exhausted BFS! uh oh')
            # pop one off, graph check
            popped = positionQueue.pop(0)
            if popped in seen:
                continue
            seen.add(popped)

            x, y = popped[0], popped[1]
            x = int(x)
            y = int(y)
            if (allGood(state, x, y)):
                state.data.food[x][y] = True
                foodAdded.append((x, y))
                numToDump -= 1

            # generate successors
            positionQueue = positionQueue + genSuccessors(x, y)

        state.data._foodAdded = foodAdded
        # now our agentState is no longer carrying food
        agentState.numCarrying = 0
        pass

    dumpFoodFromDeath = staticmethod(dumpFoodFromDeath)

    def checkDeath( state, agentIndex):
        agentState = state.data.agentStates[agentIndex]
        if state.isOnRedTeam(agentIndex):
            otherTeam = state.getBlueTeamIndices()
        else:
            otherTeam = state.getRedTeamIndices()
        if agentState.isPacman:
            for index in otherTeam:
                otherAgentState = state.data.agentStates[index]
                if otherAgentState.isPacman: continue
                ghostPosition = otherAgentState.getPosition()
                if ghostPosition == None: continue
                if manhattanDistance( ghostPosition, agentState.getPosition() ) <= COLLISION_TOLERANCE:
                    # award points to the other team for killing Pacmen
                    if otherAgentState.scaredTimer <= 0:
                        AgentRules.dumpFoodFromDeath(state, agentState, agentIndex)

                        score = KILL_POINTS
                        if state.isOnRedTeam(agentIndex):
                            score = -score
                        state.data.scoreChange += score
                        agentState.isPacman = False
                        agentState.configuration = agentState.start
                        agentState.scaredTimer = 0
                    else:
                        score = KILL_POINTS
                        if state.isOnRedTeam(agentIndex):
                            score = -score
                        state.data.scoreChange += score
                        otherAgentState.isPacman = False
                        otherAgentState.configuration = otherAgentState.start
                        otherAgentState.scaredTimer = 0
        else: # Agent is a ghost
            for index in otherTeam:
                otherAgentState = state.data.agentStates[index]
                if not otherAgentState.isPacman: continue
                pacPos = otherAgentState.getPosition()
                if pacPos == None: continue
                if manhattanDistance( pacPos, agentState.getPosition() ) <= COLLISION_TOLERANCE:
                    #award points to the other team for killing Pacmen
                    if agentState.scaredTimer <= 0:
                        AgentRules.dumpFoodFromDeath(state, otherAgentState, agentIndex)

                        score = KILL_POINTS
                        if not state.isOnRedTeam(agentIndex):
                            score = -score
                        state.data.scoreChange += score
                        otherAgentState.isPacman = False
                        otherAgentState.configuration = otherAgentState.start
                        otherAgentState.scaredTimer = 0
                    else:
                        score = KILL_POINTS
                        if state.isOnRedTeam(agentIndex):
                            score = -score
                        state.data.scoreChange += score
                        agentState.isPacman = False
                        agentState.configuration = agentState.start
                        agentState.scaredTimer = 0
    checkDeath = staticmethod( checkDeath )

    def placeGhost(state, ghostState):
        ghostState.configuration = ghostState.start
    placeGhost = staticmethod( placeGhost )

#############################
# FRAMEWORK TO START A GAME #
#     开始游戏的框架         #
#############################

def default(str):
    return str + ' [Default: %default]'

def parseAgentArgs(str):    # 解析agent参数,并返回一个包含参数键值对的字典。
    if str == None or str == '': return {}
    pieces = str.split(',')
    opts = {}
    for p in pieces:
        if '=' in p:
            key, val = p.split('=')
        else:
            key,val = p, 1
        opts[key] = val
    return opts

def readCommand( argv ):
    """
    Processes the command used to run pacman from the command line.
    处理从命令行运行 pacman 的命令。
    """
    from optparse import OptionParser
    usageStr = """
    USAGE:      python pacman.py <options>
    EXAMPLES:   (1) python capture.py
                                    - starts a game with two baseline agents    使用两个baseline代理启动游戏 
                            (2) python capture.py --keys0
                                    - starts a two-player interactive game where the arrow keys control agent 0, and all other agents are baseline agents
                                    开始一个双人互动游戏，其中方向键控制agent 0，所有其他代理都是baseline代理
                            (3) python capture.py -r baselineTeam -b myTeam
                                    - starts a fully automated game where the red team is a baseline team and blue team is myTeam   开始一场全自动游戏，其中红队是baseline队，蓝队是myTeam
    """
    parser = OptionParser(usageStr)

    parser.add_option('-r', '--red', help=default('Red team'),
                                        default='baselineTeam')
    parser.add_option('-b', '--blue', help=default('Blue team'),
                                        default='baselineTeam')
    parser.add_option('--red-name', help=default('Red team name'),
                                        default='Red')
    parser.add_option('--blue-name', help=default('Blue team name'),
                                        default='Blue')
    parser.add_option('--redOpts', help=default('Options for red team (e.g. first=keys)'),
                                        default='')
    parser.add_option('--blueOpts', help=default('Options for blue team (e.g. first=keys)'),
                                        default='')
    parser.add_option('--keys0', help='Make agent 0 (first red player) a keyboard agent', action='store_true',default=False)
    parser.add_option('--keys1', help='Make agent 1 (second red player) a keyboard agent', action='store_true',default=False)
    parser.add_option('--keys2', help='Make agent 2 (first blue player) a keyboard agent', action='store_true',default=False)
    parser.add_option('--keys3', help='Make agent 3 (second blue player) a keyboard agent', action='store_true',default=False)
    parser.add_option('-l', '--layout', default=0,
                                        help=default('1. Defalut: generate map randomly\n' +
                                                     '2. Specific path: The LAYOUT_FILE folder specifies the directory from which to load the map layout, for instance, layout/eval'),
                                        )
    parser.add_option('-t', '--textgraphics', action='store_true', dest='textgraphics',
                                        help='Display output as text only', default=False)

    parser.add_option('-q', '--quiet', action='store_true',
                                        help='Display minimal output and no graphics', default=False)

    parser.add_option('-Q', '--super-quiet', action='store_true', dest="super_quiet",
                                        help='Same as -q but agent output is also suppressed', default=False)

    parser.add_option('-z', '--zoom', type='float', dest='zoom',
                                        help=default('Zoom in the graphics'), default=1)
    parser.add_option('-i', '--time', type='int', dest='time',
                                        help=default('TIME limit of a game in moves'), default=1200, metavar='TIME')
    parser.add_option('-n', '--numGames', type='int',
                                        help=default('Number of games to play'), default=1)
    parser.add_option('-f', '--fixRandomSeed', action='store_true',
                                        help='Fixes the random seed to always play the same game', default=False)
    parser.add_option('--record', action='store_true',
                                        help='Writes game histories to a file (named by the time they were played)', default=False)
    parser.add_option('--replay', default=None,
                                        help='Replays a recorded game file.')
    parser.add_option('-x', '--numTraining', dest='numTraining', type='int',
                                        help=default('How many episodes are training (suppresses output)'), default=0)
    parser.add_option('-c', '--catchExceptions', action='store_true', default=False,
                                        help='Catch exceptions and enforce time limits')

    options, otherjunk = parser.parse_args(argv)
    assert len(otherjunk) == 0, "Unrecognized options: " + str(otherjunk)
    args = dict()

    # Choose a display format
    #if options.pygame:
    #   import pygameDisplay
    #    args['display'] = pygameDisplay.PacmanGraphics()
    if options.textgraphics:
        import textDisplay
        args['display'] = textDisplay.PacmanGraphics()
    elif options.quiet:
        import textDisplay
        args['display'] = textDisplay.NullGraphics()
    elif options.super_quiet:
        import textDisplay
        args['display'] = textDisplay.NullGraphics()
        args['muteAgents'] = True
    else:
        import captureGraphicsDisplay
        # Hack for agents writing to the display
        captureGraphicsDisplay.FRAME_TIME = 0
        args['display'] = captureGraphicsDisplay.PacmanGraphics(options.red, options.blue, options.zoom, 0, capture=True)
        import __main__
        __main__.__dict__['_display'] = args['display']


    args['redTeamName'] = options.red_name
    args['blueTeamName'] = options.blue_name

    if options.fixRandomSeed: random.seed('cs188')

    # Special case: recorded games don't use the runGames method or args structure
    if options.replay != None:
        print('Replaying recorded game %s.' % options.replay)
        import pickle
        recorded = pickle.load(open(options.replay))
        recorded['display'] = args['display']
        replayGame(**recorded)
        sys.exit(0)

    # Choose a pacman agent
    redArgs, blueArgs = parseAgentArgs(options.redOpts), parseAgentArgs(options.blueOpts)
    if options.numTraining > 0:
        redArgs['numTraining'] = options.numTraining
        blueArgs['numTraining'] = options.numTraining
    nokeyboard = options.textgraphics or options.quiet or options.numTraining > 0
    print('\nRed team %s with %s:' % (options.red, redArgs))
    redAgents = loadAgents(True, options.red, nokeyboard, redArgs)
    print('\nBlue team %s with %s:' % (options.blue, blueArgs))
    blueAgents = loadAgents(False, options.blue, nokeyboard, blueArgs)
    args['agents'] = sum([list(el) for el in zip(redAgents, blueAgents)],[]) # list of agents

    numKeyboardAgents = 0
    for index, val in enumerate([options.keys0, options.keys1, options.keys2, options.keys3]):
        if not val: continue
        if numKeyboardAgents == 0:
            agent = keyboardAgents.KeyboardAgent(index)
        elif numKeyboardAgents == 1:
            agent = keyboardAgents.KeyboardAgent2(index)
        else:
            raise Exception('Max of two keyboard agents supported')
        numKeyboardAgents += 1
        args['agents'][index] = agent
    random.seed(random.randint(1,10086))
    # Choose a layout
    if options.layout != 0:
        layout_path = options.layout
        if not os.path.exists(layout_path):
            raise Exception(f"The layout folder {layout_path} cannot be found")
        else:
            layout_files = glob(f'{layout_path}/*.lay')
            if layout_files is []:
                raise Exception(f"The layout folder {layout_path} is empty, " + 
                                "please use utils/layout_generator.py to generate the layout files")
        layouts = [layout.getLayout(val) for val in 
                   random.sample(layout_files, options.numGames)]
    else:
        seed_map= random.sample(range(100),options.numGames)
        print(f"Map seed {*seed_map,}")
        layouts = [layout.Layout(generateMaze(int(idx)).split('\n')) for idx in seed_map]
    args['layouts'] = layouts
    args['length'] = options.time
    args['numGames'] = options.numGames
    args['numTraining'] = options.numTraining
    args['record'] = options.record
    args['catchExceptions'] = options.catchExceptions
    return args



import traceback   # 用于提供详细的异常信息
def loadAgents(isRed, factory, textgraphics, cmdLineArgs):
    "Calls agent factories and returns lists of agents"
    # 调用agent factory并返回agent列表
    try:
        if not factory.endswith(".py"):
            factory += ".py"
        print(('player' + str(int(isRed)), factory))
        module = importlib.machinery.SourceFileLoader('player' + str(int(isRed)), factory).load_module()
    except (NameError, ImportError):
        print('Error: The team "' + factory + '" could not be loaded! ', file=sys.stderr)
        traceback.print_exc()
        return [None for i in range(2)]

    args = dict()
    args.update(cmdLineArgs)  # Add command line args with priority 添加具有优先级的命令行参数

    print("Loading Team:", factory)
    print("Arguments:", args)

    # if textgraphics and factoryClassName.startswith('Keyboard'):
    #   raise Exception('Using the keyboard requires graphics (no text display, quiet or training games)')

    try:
        createTeamFunc = getattr(module, 'createTeam')
    except AttributeError:
        print('Error: The team "' + factory + '" could not be loaded! ', file=sys.stderr)
        traceback.print_exc()
        return [None for i in range(2)]

    indexAddend = 0
    if not isRed:
        indexAddend = 1
    indices = [2*i + indexAddend for i in range(2)]
    return createTeamFunc(indices[0], indices[1], isRed, **args)

def replayGame( layout, agents, actions, display, length, redTeamName, blueTeamName ):
        rules = CaptureRules()
        game = rules.newGame( layout, agents, display, length, False, False )
        state = game.state
        display.redTeam = redTeamName
        display.blueTeam = blueTeamName
        display.initialize(state.data)

        for action in actions:
            # Execute the action
            state = state.generateSuccessor( *action )
            # Change the display
            display.update( state.data )
            # Allow for game specific conditions (winning, losing, etc.)
            rules.process(state, game)

        display.finish()

def runGames( layouts, agents, display, length, numGames, record, numTraining, redTeamName, blueTeamName, muteAgents=False, catchExceptions=False ):

    rules = CaptureRules()  # CaptureRules实例
    games = []          # 空列表games来存储游戏的结果。

    if numTraining > 0:
        print('Playing %d training games' % numTraining)    # 判断是否进行训练游戏，并打印相关信息。

    for i in range( numGames ):
        beQuiet = i < numTraining
        layout = layouts[i]
        if beQuiet: # 如果游戏是训练游戏（即beQuiet=True）,这里指的是使用指令-q或-Q
                # Suppress output and graphics
                import textDisplay
                gameDisplay = textDisplay.NullGraphics()
                rules.quiet = True
        else: # 如果游戏不是训练游戏（即beQuiet=False）
                gameDisplay = display
                rules.quiet = False
        g = rules.newGame( layout, agents, gameDisplay, length, muteAgents, catchExceptions )
        g.run()
        if not beQuiet: games.append(g)

        g.record = None
        if record:
            import time, pickle, game
            #fname = ('recorded-game-%d' % (i + 1)) +  '-'.join([str(t) for t in time.localtime()[1:6]])
            #f = file(fname, 'w')
            components = {'layout': layout, 'agents': [game.Agent() for a in agents], 'actions': g.moveHistory, 'length': length, 'redTeamName': redTeamName, 'blueTeamName':blueTeamName }
            #f.close()
            print("recorded")
            g.record = pickle.dumps(components)
            with open('replay-%d'%i,'wb') as f:
                f.write(g.record)

    if numGames > 1:    # 如果有多个游戏，则会打印游戏的平均分数、得分、红队和蓝队的胜率，以及记录（胜方或平局）统计信息。
        scores = [game.state.data.score for game in games]
        redWinRate = [s > 0 for s in scores].count(True)/ float(len(scores))
        blueWinRate = [s < 0 for s in scores].count(True)/ float(len(scores))
        print('Average Score:', sum(scores) / float(len(scores)))
        print('Scores:       ', ', '.join([str(score) for score in scores]))
        print('Red Win Rate:  %d/%d (%.2f)' % ([s > 0 for s in scores].count(True), len(scores), redWinRate))
        print('Blue Win Rate: %d/%d (%.2f)' % ([s < 0 for s in scores].count(True), len(scores), blueWinRate))
        print('Record:       ', ', '.join([('Blue', 'Tie', 'Red')[max(0, min(2, 1 + s))] for s in scores]))
    return games

def save_score(game):
    with open('score', 'w') as f:
        print(game.state.data.score, file=f)

if __name__ == '__main__':
    """
    The main function called when pacman.py is run
    from the command line:

    > python capture.py

    See the usage string for more details.

    > python capture.py --help
    """
    options = readCommand(sys.argv[1:]) # Get game components based on input    根据输入获取游戏组件
    games = runGames(**options)

    # save_score(games[0])