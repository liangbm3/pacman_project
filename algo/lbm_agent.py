# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint


##########
# Agents #
##########

class OffensiveAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        if self.red:
            CaptureAgent.registerTeam(self, gameState.getRedTeamIndices())
        else:
            CaptureAgent.registerTeam(self, gameState.getBlueTeamIndices())
        #初始化状态量
        self.attacker_state="C"
        self.own_death_signal=False
        self.enemy_death_signal=False
        self.start_position=gameState.getAgentPosition(self.index)
        
    def chooseAction(self, gameState):
        self.judge_state(gameState)
    def judge_state(self,gameState):
        if len(self.getEnemyPos(gameState=gameState))!=0:
            self.attacker_state="A"
        elif 
        
    
    #检测敌方的位置，如果敌方可以被看见，则返回一个元组（敌方索引，位置）的列表，否则返回空列表
    def getEnemyPos(self, gameState):
        enemyPos = []
        for enemy in self.getOpponents(gameState):
            pos = gameState.getAgentPosition(enemy)
            # Will need inference if None
            if pos != None:
                enemyPos.append((enemy, pos))
        return enemyPos
    
    #返回离自己最近敌人的距离
    def enemyDist(self, gameState):
        pos = self.getEnemyPos(gameState)
        minDist = None
        if len(pos) > 0:
            minDist = float('inf')
            myPos = gameState.getAgentPosition(self.index)
            for i, p in pos:
                dist = self.getMazeDistance(p, myPos)
                if dist < minDist:
                    minDist = dist
        return minDist
            
        
