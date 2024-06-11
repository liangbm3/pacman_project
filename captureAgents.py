# captureAgents.py
# ----------------
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


# ---------------------------------------------------------------------------- #
#这个文件包含基本的智能体类CaptureAgent，可以通过继承它并重构某些函数来创建自己的智能体
# ---------------------------------------------------------------------------- #

"""
    Interfaces for capture agents and agent factories
    捕获agent接口和agent factories接口"     ??? 什么是factory
"""

from game import Agent
import distanceCalculator
from util import nearestPoint
import util
import random

# Note: the following class is not used, but is kept for backwards
# compatibility with team submissions that try to import it.

# ---------------------------------------------------------------------------- #
#AgentFactory类，用来创建agent？
# ---------------------------------------------------------------------------- #
class AgentFactory:
    "Generates agents for a side（创建一个agent）"

    def __init__(self, isRed, **args):
        self.isRed = isRed

    def getAgent(self, index):
        "Returns the agent for the provided index.（返回是哪个agent）"
        util.raiseNotDefined()  #该函数的主要目的是用于标记尚未实现的方法，以便在代码中进行标记和调试。
                                #当程序尝试调用尚未实现的方法时，会打印出相应的错误消息，并终止程序的执行

# ---------------------------------------------------------------------------- #
#RandomAgent类，创建一个随机Agent
# ---------------------------------------------------------------------------- #
class RandomAgent( Agent ):
    """
    A random agent that abides by the rules.
    根据规则创建一个随机agent
    """
    def __init__( self, index ):
        self.index = index

    def getAction( self, state ):
        return random.choice( state.getLegalActions( self.index ) )

class CaptureAgent(Agent):
    """
    A base class for capture agents.  The convenience methods herein handle
    some of the complications of a two-team game.
    CaptureAgent的基类。此处的便捷方法可处理双队游戏中的一些复杂情况。
    
    Recommended Usage:  Subclass CaptureAgent and override chooseAction.
    建议用法：子类化 CaptureAgent 并覆盖 chooseAction。
    """

    #############################
    # Methods to store key info
    # 存储关键信息的方法     #
    #############################

    def __init__( self, index, timeForComputing = .1 ):
        """
        Lists several variables you can query:
        self.index = index for this agent（agent的索引）
        self.red = true if you're on the red team, false otherwise（是否为红队）
        self.agentsOnTeam = a list of agent objects that make up your team（组成团队的列表对象）
        self.distancer = distance calculator (contest code provides this)（距离计算器）
        self.observationHistory = list of GameState objects that correspond
                to the sequential order of states that have occurred so far this game
                （与本游戏迄今为止发生的状态的顺序相对应的 GameState 对象列表）
        self.timeForComputing = an amount of time to give each turn for computing maze distances
                (part of the provided distance calculator)
                （计算迷宫距离时每回合给出的时间量，提供的距离计算器的一部分）
        """
        # Agent index for querying state    
        self.index = index

        # Whether or not you're on the red team
        self.red = None

        # Agent objects controlling you and your teammates
        self.agentsOnTeam = None

        # Maze distance calculator
        self.distancer = None

        # A history of observations
        self.observationHistory = []

        # Time to spend each turn on computing maze distances
        self.timeForComputing = timeForComputing

        # Access to the graphics
        self.display = None

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)
        仅在Agent/游戏初始化时调用一次， 
        初始化地图信息、 初始化distanceCalculator.Distancer最短距离测算子， 
        以及计算Agent与地图相关元素的初始距离
        """
        self.red = gameState.isOnRedTeam(self.index)
        self.distancer = distanceCalculator.Distancer(gameState.data.layout)

        # comment this out to forgo maze distance computation and use manhattan distances
        self.distancer.getMazeDistances()

        import __main__
        if '_display' in dir(__main__):
            self.display = __main__._display

    def final(self, gameState):
        self.observationHistory = []

    def registerTeam(self, agentsOnTeam):
        """
        Fills the self.agentsOnTeam field with a list of the
        indices of the agents on your team.
        使用提供的团队中agent的索引列表来填充 self.agentsOnTeam 
        """
        self.agentsOnTeam = agentsOnTeam

    def observationFunction(self, gameState):
        " Changing this won't affect pacclient.py, but will affect capture.py "
        # 更改此项不会影响 pacclient.py，但会影响 capture.py
        return gameState.makeObservation(self.index)

    def debugDraw(self, cells, color, clear=False):

        if self.display:
            from captureGraphicsDisplay import PacmanGraphics
            if isinstance(self.display, PacmanGraphics):
                if not type(cells) is list:
                    cells = [cells]
                self.display.debugDraw(cells, color, clear)

    def debugClear(self):
        if self.display:
            from captureGraphicsDisplay import PacmanGraphics
            if isinstance(self.display, PacmanGraphics):
                self.display.clearDebug()

    #################
    # Action Choice #
    #################

    def getAction(self, gameState):
        """
        Calls chooseAction on a grid position, but continues on half positions.
        If you subclass CaptureAgent, you shouldn't need to override this method.  It
        takes care of appending the current gameState on to your observation history
        (so you have a record of the game states of the game) and will call your
        choose action method if you're in a state (rather than halfway through your last
        move - this occurs because Pacman agents move half as quickly as ghost agents).
        
        在网格位置调用 chooseAction，但在半个位置继续。
        如果您将 CaptureAgent 子类化，则无需重写此方法。
        它负责将当前游戏状态附加到您的观察历史记录中（这样您就有了游戏状态的记录），
        并且如果您处于某个状态（而不是上次移动的一半 - 发生这种情况是因为 Pacman agent的移动速度是幽灵agent的一半），
        它将调用您的选择操作方法。
        """
        self.observationHistory.append(gameState)

        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        if myPos != nearestPoint(myPos):
            # We're halfway from one position to the next
            return gameState.getLegalActions(self.index)[0]
        else:
            return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        """
        Override this method to make a good agent. It should return a legal action within
        the time limit (otherwise a random legal action will be chosen for you).
        重写此方法可制作一个好的agent。它应在时限内返回合法操作（否则将为您随机选择合法操作）。
        """
        util.raiseNotDefined()

    #######################
    # Convenience Methods #
    #######################

    def getFood(self, gameState):
        """
        Returns the food you're meant to eat. This is in the form of a matrix
        where m[x][y]=true if there is food you can eat (based on your team) in that square.
        返回您要吃的食物。它以矩阵的形式显示，其中如果该方格中有您可以吃的食物（根据您的团队），则 m[x][y]=true。
        """
        if self.red:
            return gameState.getBlueFood()
        else:
            return gameState.getRedFood()

    def getFoodYouAreDefending(self, gameState):
        """
        Returns the food you're meant to protect (i.e., that your opponent is
        supposed to eat). This is in the form of a matrix where m[x][y]=true if
        there is food at (x,y) that your opponent can eat.
        返回您要保护的食物（即您的对手应该吃的食物）。它以矩阵的形式出现，
        其中如果 (x,y) 处有您的对手可以吃的食物，则 m[x][y]=true。
        """
        if self.red:
            return gameState.getRedFood()
        else:
            return gameState.getBlueFood()

    def getCapsules(self, gameState):
        if self.red:
            return gameState.getBlueCapsules()
        else:
            return gameState.getRedCapsules()

    def getCapsulesYouAreDefending(self, gameState):
        if self.red:
            return gameState.getRedCapsules()
        else:
            return gameState.getBlueCapsules()

    def getOpponents(self, gameState):
        """
        Returns agent indices of your opponents. This is the list of the numbers
        of the agents (e.g., red might be "1,3,5")
        返回您对手的索引列表。这是一个数字列表，例如红队可能是 "1,3,5"
        """
        if self.red:
            return gameState.getBlueTeamIndices()
        else:
            return gameState.getRedTeamIndices()

    def getTeam(self, gameState):
        """
        Returns agent indices of your team. This is the list of the numbers
        of the agents (e.g., red might be the list of 1,3,5)
        返回您的队伍的索引列表。这是一个数字列表，例如红队可能是 1,3,5 列表
        """
        if self.red:
            return gameState.getRedTeamIndices()
        else:
            return gameState.getBlueTeamIndices()

    def getScore(self, gameState):
        """
        Returns how much you are beating the other team by in the form of a number
        that is the difference between your score and the opponents score.  This number
        is negative if you're losing.
        返回您赢得其他队伍的分数的形式。这是一个数字，它是您得分与对手得分之间的差值。
        """
        if self.red:
            return gameState.getScore()
        else:
            return gameState.getScore() * -1

    def getMazeDistance(self, pos1, pos2):
        """
        Returns the distance between two points; These are calculated using the provided
        distancer object.
        返回两个点之间的距离。这些是使用提供的 distancer 对象计算的。
        If distancer.getMazeDistances() has been called, then maze distances are available.
        Otherwise, this just returns Manhattan distance.
        如果 distancer.getMazeDistances() 已被调用，则可获得迷宫距离。
        否则，它只返回曼哈顿距离。
        """
        d = self.distancer.getDistance(pos1, pos2)
        return d

    def getPreviousObservation(self):
        """
        Returns the GameState object corresponding to the last state this agent saw
        (the observed state of the game last time this agent moved - this may not include
        all of your opponent's agent locations exactly).
        返回上次看到该 agent 状态的 GameState 对象
        (上一次该 agent 移动时的游戏状态 - 这可能不包括所有您的对手 agent 位置的精确位置）。
        """
        if len(self.observationHistory) == 1: return None
        else: return self.observationHistory[-2]

    def getCurrentObservation(self):
        """
        Returns the GameState object corresponding this agent's current observation
        (the observed state of the game - this may not include
        all of your opponent's agent locations exactly).
        返回当前 agent 观察的 GameState 对象
        (游戏的观察状态 - 这可能不包括所有您的对手 agent 位置的精确位置)。
        """
        return self.observationHistory[-1]

    def displayDistributionsOverPositions(self, distributions):
        """
        Overlays a distribution over positions onto the pacman board that represents
        an agent's beliefs about the positions of each agent.
        叠加分布在 pacman 板上的位置，表示每个 agent 的位置分布的 agent 信念。
        The arg distributions is a tuple or list of util.Counter objects, where the i'th
        Counter has keys that are board positions (x,y) and values that encode the probability
        that agent i is at (x,y).
        该参数 distributions 是一个元组或列表，其中第 i 个 Counter 对象有键值对，
        其中键是 board 位置 (x,y)，值是编码 agent i 位于 (x,y) 位置的概率。
        If some elements are None, then they will be ignored.  If a Counter is passed to this
        function, it will be displayed. This is helpful for figuring out if your agent is doing
        inference correctly, and does not affect gameplay.
        如果某些元素为 None，则将忽略它们。如果传递了一个 Counter 对象到该函数中，则会显示。
        这对于检查 agent 是否正确推理非常有用，不会影响游戏。
        """
        dists = []
        for dist in distributions:  #遍历输入的分布数据
            if dist != None:
                if not isinstance(dist, util.Counter): raise Exception("Wrong type of distribution")
                dists.append(dist)
            else:
                dists.append(util.Counter())    
        if self.display != None and 'updateDistributions' in dir(self.display):
            self.display.updateDistributions(dists)
        else:
            self._distributions = dists # These can be read by pacclient.py


class TimeoutAgent( Agent ):
    """
    A random agent that takes too much time. Taking
    too much time results in penalties and random moves.
    一个随机agent，耗时过多。耗时过多会导致处罚和随机移动。
    """
    def __init__( self, index ):
        self.index = index

    def getAction( self, state ):
        import random, time
        time.sleep(2.0)
        return random.choice( state.getLegalActions( self.index ) )
