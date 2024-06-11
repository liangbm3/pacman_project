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
from baselineTeam import OffensiveReflexAgent   # 我自己加的,把secend = 'agent'换成了OffensiveReflexAgent
import random, time, util
from game import Directions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
                             first = 'DummyAgent', second = 'OffensiveReflexAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.
    此函数应返回将组成团队的两个agent的列表，使用 firstIndex 和 secondIndex 作为其agent索引号进行初始化。
    如果正在创建红队，则 isRed 为 True，如果正在创建蓝队，则为 False。

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    作为一种可能有用的开发辅助工具，此函数可以采用额外的字符串值关键字参数（此函数中的“first”和“second”就是此类参数），
    这些参数来自 --redOpts 和 --blueOpts 命令行参数到 capture.py。
    但是，对于nightly比赛，您的团队将在没有任何额外参数的情况下创建，因此您应该确保默认行为是您想要的nightly比赛行为。
    (什么是nightly contest？)
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
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
        此方法处理agent的初始设置，以填充有用的字段（例如我们所在的团队）。

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        distanceCalculator 实例缓存每对位置之间的迷宫距离，
        因此您的agent可以使用：self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        重要提示：此方法最多可运行 15 秒。  (??? 15秒是什么意思,处理初始设置的时间只有15秒吗?)
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.

        确保不要删除以下行。如果您想使用曼哈顿距离而不是迷宫距离以节省初始化时间，请查看captureAgents.py中的CaptureAgent.registerInitialState。
        (所以怎么使用曼哈顿距离节省初始时间?)
        '''
        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        如果需要的话，您的初始化代码就放在这里。
        '''


    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        这个方法是随机选择动作。(没什么用,到时肯定不能用这个)
        """
        actions = gameState.getLegalActions(self.index)

        '''
        You should change this in your own agent.
        您应该在您自己的agent中更改这一点。
        '''

        return random.choice(actions)

