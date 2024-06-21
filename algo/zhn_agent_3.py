
from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
from game import Grid
import math

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'OffensiveReflexAgent'):
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
  walls = gameState.getWalls()
  x, y = pos
  if isRed:
    return 1.0 * x / walls.width
  else:
    return 1.0 * (walls.width - x) / walls.width

def getDeepFood(gameState, agent, isRed):
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
  if len(capsules) == 0:
    return 0
  minDist = 1000
  for capsule in capsules:
    dist = agent.getMazeDistance(pos, capsule)
    if dist < minDist:
      minDist = dist        
  return minDist

def getCloseSafePoints(gameState, isRed):
  walls = gameState.getWalls()
  if isRed:
    col = int((walls.width / 2)) - 1
  else:
    col = int((walls.width / 2)) 
  safePoints = []
  for y in range(walls.height):
    if not walls[col][y]:
      safePoints.append((col,y))
  return safePoints

def getPositionAfterAction(pos, action):
  x,y = pos
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
  x,y = pos
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

# Find all points with just one possible action -> Dead ends
# Travel along dead ends until an opening is found -> choke point
# Find depths of positions from prior movement

def getDeadEnds(gameState):
  walls = gameState.getWalls()
  deadEnds = []
  for r in range(walls.height):
    for c in range(walls.width):
      pos = (c,r) 
      if not walls[c][r]:
        if len(getPossibleActions(gameState, pos)) <= 2: #Stop and one other action
          deadEnds.append(pos)
  return deadEnds

def getChokePointAndDirection(gameState, deadEnd):
  walls = gameState.getWalls()
  pos = deadEnd
  actions = getPossibleActions(gameState, pos)
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
  
def getChokePointAndDirectionRestricted(gameState, deadEnd, directionsRestricted):
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
  walls = gameState.getWalls()
  deadEnds = getDeadEnds(gameState)
  chokePointList = []
  chokePoints = {}
  toCheck = {}
  stillGoing = False
  for deadEnd in deadEnds:
    chokePoint, direction = getChokePointAndDirection(gameState, deadEnd)
    if not chokePoint in chokePoints:
      chokePoints[chokePoint] = []   
    chokePoints[chokePoint].append(direction)
    if len(getPossibleActions(gameState, chokePoint)) - 2 == len(chokePoints[chokePoint]):
      toCheck[chokePoint] = chokePoints[chokePoint]
      del chokePoints[chokePoint]
      stillGoing = True
  for chokePoint in chokePoints:
    chokePointList.append((chokePoint, chokePoints[chokePoint]))

  while stillGoing: 
    deadEnds = toCheck.copy()
    toCheck = {}
    chokePoints = {}
    stillGoing = False
    for deadEnd in deadEnds:
      chokePoint, direction = getChokePointAndDirectionRestricted(gameState, deadEnd, deadEnds[deadEnd])
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
  minDist = 1000
  walls = gameState.getWalls()
  for choke, dirs in chokePoints:
    dist = agent.getMazeDistance(pos, choke)
    if dist < minDist:
      minDist = dist
  return minDist

def bfsDepthGrid(gameState, chokePoint, direction, depthGrid):
  x,y = chokePoint
  depthGrid[x][y] = 9
  newPos = getPositionAfterAction(chokePoint, direction)
  posQueue = util.Queue()
  posQueue.push((newPos, 1))
  while not posQueue.isEmpty():
    pos, depth = posQueue.pop()
    x,y = pos
    if depthGrid[x][y] == 0:
      depthGrid[x][y] = depth   
      actions = getPossibleActions(gameState, pos)
      for action in actions:
        nextPos = getPositionAfterAction(pos, action)
        newDepth = depth + 1
        posQueue.push((nextPos, newDepth))
  x,y = chokePoint
  depthGrid[x][y] = 0
  return depthGrid

def getDepthsAndChokePoints(gameState):
  walls = gameState.getWalls()
  depthGrid = Grid(walls.width, walls.height, initialValue=0)
  
  chokePoints = getChokePoints(gameState)
  for chokePoint, directions in chokePoints:
    for direction in directions:
      depthGrid = bfsDepthGrid(gameState, chokePoint, direction, depthGrid)
  return depthGrid, chokePoints

def getDistToSafety(gameState, agent):
  agentIndex = agent.index
  isPacman = gameState.getAgentState(agentIndex).isPacman
  #if not isPacman:
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
  agentIndex = agent.index
  isRed = agent.red
  pos = gameState.getAgentPosition(agentIndex)
  minAdvantage = 1000
  if isDefending:
    closestSafePoints = agent.getCapsulesYouAreDefending(gameState)
    minAdvantage = 1000
  else:
    closestSafePoints = agent.getCapsules(gameState)
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

enemyLocationPredictor = None
PERSIST = 10

class EnemyLocationPredictor:
  
  

  def __init__(self, gameState, agent):
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
      x,y = int(x), int(y)
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
    x,y = pos

    if (x < (self.walls.width / 2) and isRed) or (x >= (self.walls.width / 2) and not isRed):
      #Add position to avoid

      self.ignoreCounterAvoid = PERSIST 

      if not pos in self.positionsToAvoid:
        if len(self.positionsToAvoid) > defenders:
          while len(self.positionsToAvoid) >= defenders and len(self.positionsToAvoid) > 0:
            self.positionsToAvoid.pop(0)
        self.positionsToAvoid.append(pos)

    else:
      #Add position to investigate

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

  def updatePart(self, gameState, agent, teamPosition1, teamPosition2, verbose = False):
    invaders = len([a for a in self.enemies if gameState.getAgentState(a).isPacman])
    defenders = len([a for a in self.enemies if not gameState.getAgentState(a).isPacman])
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
      knownPos1 = gameState.getAgentState(self.enemyIndices[0]).getPosition()
    else:
      for r in range(self.walls.height):
        for c in range(self.walls.width):
          manhattanDist1 = abs(c - team1X) + abs(r - team1Y)
          manhattanDist2 = abs(c - team2X) + abs(r - team2Y)
          if min(manhattanDist1, manhattanDist2) <= 5:
            possiblePositions1[c][r] = False

    if not gameState.getAgentState(self.enemyIndices[1]).getPosition() is None:
      knownPos2 = gameState.getAgentState(self.enemyIndices[1]).getPosition()
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
      dist = min(agent.getMazeDistance(self.enemy1KnownLocation, self.pastTeamLocation1), agent.getMazeDistance(self.enemy1KnownLocation, self.pastTeamLocation2))
      if dist < 3:
        enemy1Captured = True
      else:
        enemy1Captured = False
    else:
      enemy1Captured = False

    if not self.enemy2KnownLocation is None and knownPos2 is None:
      dist = min(agent.getMazeDistance(self.enemy2KnownLocation, self.pastTeamLocation1), agent.getMazeDistance(self.enemy2KnownLocation, self.pastTeamLocation2))
      if dist < 3:
        enemy2Captured = True
      else:
        enemy2Captured = False
    else:
      enemy2Captured = False

    if enemy1Captured:
      self.addPositionForConsideration(self.enemyStartPos, invaders, defenders, isRed)
      knownPos1 = self.enemyStartPos
    if enemy2Captured:
      self.addPositionForConsideration(self.enemyStartPos, invaders, defenders, isRed)
      knownPos2 = self.enemyStartPos

    # If a team member was captured, assume that the nearest enemy to that team member is now where the team member was. (POS NOT KNOWN)
    # An enemy can't capture a team member without the team member knowing where the enemy is at the time of the capture
    if team1Captured and team2Captured:
      if not self.enemy1KnownLocation is None and not self.enemy2KnownLocation is None:

        dist11 = agent.getMazeDistance(self.enemy1KnownLocation, self.pastTeamLocation1)
        dist12 = agent.getMazeDistance(self.enemy1KnownLocation, self.pastTeamLocation2)
        dist21 = agent.getMazeDistance(self.enemy2KnownLocation, self.pastTeamLocation1)
        dist22 = agent.getMazeDistance(self.enemy2KnownLocation, self.pastTeamLocation2)

        canBothCapture1 = False
        canBothCapture2 = False

        if dist11 < 3 and dist21 < 3:
          #Both agents could capture team1
          canBothCapture1 = True
        elif dist11 < 3 and knownPos1 is None:
          #Enemy1 must have captured team1
          knownPos1 = self.pastTeamLocation1
          self.addPositionForConsideration(self.pastTeamLocation1, invaders, defenders, isRed)
        elif dist21 < 3 and knownPos2 is None:
          #Enemy2 must have captured team1
          knownPos2 = self.pastTeamLocation1
          self.addPositionForConsideration(self.pastTeamLocation1, invaders, defenders, isRed)

        if dist12 < 3 and dist22 < 3:
          #Both agents could capture team2
          canBothCapture2 = True
        elif dist12 < 3 and knownPos1 is None:
          #Enemy1 must have captured team2
          knownPos1 = self.pastTeamLocation2
          self.addPositionForConsideration(self.pastTeamLocation2, invaders, defenders, isRed)
        elif dist22 < 3 and knownPos2 is None:
          #Enemy2 must have captured team2
          knownPos2 = self.pastTeamLocation2
          self.addPositionForConsideration(self.pastTeamLocation2, invaders, defenders, isRed)
        
        if canBothCapture1 and canBothCapture2:
          #Make arbitrary assumption
          self.addPositionForConsideration(self.pastTeamLocation1, invaders, defenders, isRed)
          self.addPositionForConsideration(self.pastTeamLocation2, invaders, defenders, isRed)
        elif canBothCapture1:
          self.addPositionForConsideration(self.pastTeamLocation1, invaders, defenders, isRed)
        elif canBothCapture2:
          self.addPositionForConsideration(self.pastTeamLocation2, invaders, defenders, isRed)
        

      elif not self.enemy1KnownLocation is None and knownPos1 is None:
        #Enemy1 Captured both agents at once
        knownPos1 = self.pastTeamLocation1
        self.addPositionForConsideration(self.pastTeamLocation1, invaders, defenders, isRed)
      elif knownPos2 is None:
        #Enemy2 Captured both agents at once
        knownPos2 = self.pastTeamLocation1
        self.addPositionForConsideration(self.pastTeamLocation1, invaders, defenders, isRed)
  
    elif team1Captured:
      if not self.enemy1KnownLocation is None and not self.enemy2KnownLocation is None:
        dist1 = agent.getMazeDistance(self.enemy1KnownLocation, self.pastTeamLocation1)
        dist2 = agent.getMazeDistance(self.enemy2KnownLocation, self.pastTeamLocation1)

        if dist1 < 3 and dist2 < 3:
          pass

        elif dist1 < 3 and knownPos1 is None:
          #Enemy1 must have captured team1
          knownPos1 = self.pastTeamLocation1
          self.addPositionForConsideration(self.pastTeamLocation1, invaders, defenders, isRed)

        elif knownPos2 is None:
          knownPos2 = self.pastTeamLocation1
          #Enemy2 must have captured team1
          self.addPositionForConsideration(self.pastTeamLocation1, invaders, defenders, isRed)
      
      elif not self.enemy1KnownLocation is None and knownPos1 is None:
        #Enemy1 must have captured team1
        knownPos1 = self.pastTeamLocation1
        self.addPositionForConsideration(self.pastTeamLocation1, invaders, defenders, isRed)

      elif knownPos2 is None:
        knownPos2 = self.pastTeamLocation1
        #Enemy2 must have captured team1
        self.addPositionForConsideration(self.pastTeamLocation1, invaders, defenders, isRed)

    elif team2Captured:
      if not self.enemy1KnownLocation is None and not self.enemy2KnownLocation is None:
        dist1 = agent.getMazeDistance(self.enemy1KnownLocation, self.pastTeamLocation2)
        dist2 = agent.getMazeDistance(self.enemy2KnownLocation, self.pastTeamLocation2)

        if dist1 < 3 and dist2 < 3 and knownPos1 is None and knownPos2 is None:
          #Both agents could capture team1, arbitrary decision
          self.addPositionForConsideration(self.pastTeamLocation2, invaders, defenders, isRed)

        elif dist1 < 3 and knownPos1 is None:
          #Enemy1 must have captured team1
          knownPos1 = self.pastTeamLocation2
          self.addPositionForConsideration(self.pastTeamLocation2, invaders, defenders, isRed)

        elif dist2 < 3 and knownPos2 is None:
          #Enemy2 must have captured team1
          knownPos2 = self.pastTeamLocation2
          self.addPositionForConsideration(self.pastTeamLocation2, invaders, defenders, isRed)
      
      elif not self.enemy1KnownLocation is None and knownPos1 is None:
        #Enemy1 must have captured team1
        knownPos1 = self.pastTeamLocation2
        self.addPositionForConsideration(self.pastTeamLocation2, invaders, defenders, isRed)

      elif knownPos2 is None:
        #Enemy2 must have captured team1
        knownPos2 = self.pastTeamLocation2
        self.addPositionForConsideration(self.pastTeamLocation2, invaders, defenders, isRed) 
    

    # If food was eaten, assume that the nearest enemy to that food is now where the food was. Otherwise, no enemy can be at a food location (POS NOT KNOWN)
    
    if len(self.pastFood.asList()) > len(newFood.asList()):
      for r in range(self.walls.height):
        for c in range(self.walls.width):
          if self.pastFood[c][r] and not newFood[c][r]:
            pos = (c,r)
            eatenFood.append(pos)
      #for each food eaten, get the total probability of spaces surrounding the food for both enemies.
      for pos in eatenFood:
        self.addPositionForConsideration(pos, invaders, defenders, isRed)

    
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
        self.addPositionForConsideration(pos, invaders, defenders, isRed)
    
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

  def update(self, gameState, agent, teamPosition1, teamPosition2, verbose = False):
    self.possiblePositions1, self.possiblePositions2 = self.updatePart(gameState, agent, teamPosition1, teamPosition2, verbose)

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
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    global enemyLocationPredictor
    global mode1
    global mode2
    CaptureAgent.registerInitialState(self, gameState)
    self.ignoreCounter = 0
    self.start = gameState.getAgentPosition(self.index)
    self.pastLocation = self.start
    self.depthGrid, self.chokePoints = getDepthsAndChokePoints(gameState)
    if enemyLocationPredictor == None:
      self.isLead = True
      self.targetId = 1
      enemyLocationPredictor = EnemyLocationPredictor(gameState, self)
    else:
      self.targetId = 2
      self.isLead = False

    self.prevFood = self.getFood(gameState)
    self.prevFoodDef = self.getFoodYouAreDefending(gameState)
    self.prevScore = self.getScore(gameState)
    self.prevCapsules = self.getCapsules(gameState)
    self.prevEnemyCapsules = self.getCapsules(gameState)
    
    self.openSpaces = 0
    for a in gameState.getWalls():
      for b in a:
        if not b:
          self.openSpaces += 1

    #Starting Mode
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
  def getNextMode(self, gameState, verbose = False):
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


    state = gameState.getAgentState(self.index)
    currentPos = gameState.getAgentPosition(self.index)
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    enemiesKnown = [a for a in enemies if a.getPosition() != None]
    invaders = [a for a in enemies if a.isPacman]
    invadersKnown = [a for a in enemies if a.isPacman and a.getPosition() != None]
    defenders = [a for a in enemies if not a.isPacman]
    defendersKnown = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    
    invaderKnownPositions = [a.getPosition() for a in invadersKnown]
    defenderKnownPositions = [a.getPosition() for a in defendersKnown]

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

    scoreDiff = self.getScore(gameState)

    """
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

    teamPos = gameState.getAgentPosition(self.teamIndex)
    shallowTargetsExist = False
    for shallow in getShallowFood(gameState, self, self.red).asList():
      if self.getMazeDistance(shallow, currentPos) < self.getMazeDistance(shallow, teamPos):
        shallowTargetsExist = True
        break

    if len(defendersKnown) > 0:
      capsuleAdvantage = min([getDistAdvantageCapsule(gameState, self, a.getPosition(), not state.isPacman) for a in defendersKnown])
    else:
      capsuleAdvantage = -1

    #TODO
    captured = self.justSpawned
    invaderNear = invaderDist < 6
    defenderNear = defenderDist < 4 
    isAggressive = getAggression(gameState, currentPos, self.red) >= 0.25
    significantAdvantage = scoreDiff > 8
    invadersExist = len(invaders)
    onOffense = state.isPacman
    canChargeCapsule = capsuleAdvantage >= 0
    holdingEnoughFood = self.dotsHeld >= ENOUGH_DOTS
    isCapsuleOn = self.capsuleOn
    isEnemyCapsuleOn = self.enemyCapsuleOn
    needHelp = chargeInvaderCounter > 30
    aLotOfFoodHeldByEnemies = self.dotsHeldByInvaders > 4
    invaderPresentForAWhile = invaderPresentCounter > 12

    if currentMode == "Travel to Center":
      if invaderNear and (not isEnemyCapsuleOn):
        return "Charge Invader"
      elif isAggressive:
        if significantAdvantage and (not onOffense):
          return "Sentry"
        elif shallowTargetsExist:
          return "Shallow Offense"
        else:
          return "Deep Offense"
      else:
        return "Travel to Center"

    elif currentMode == "Charge Invader":
      if captured:
        return "Travel to Center"
      elif ((invadersExist and (not invaderNear)) or ((not invadersExist) and significantAdvantage)) and (not onOffense):
        return "Sentry"
      elif (not invadersExist) and (not significantAdvantage) and shallowTargetsExist and (not significantAdvantage):
        return "Shallow Offense"
      elif (not invadersExist) and (not significantAdvantage) and (not shallowTargetsExist) and (not significantAdvantage):
        return "Deep Offense"
      else:
        return "Charge Invader"

    elif currentMode == "Retreat":
      if captured:
        return "Travel to Center"
      elif invaderNear and onOffense and canChargeCapsule and (not isCapsuleOn):
        return "Charge Capsule"
      elif not onOffense:
        return "Sentry"
      else:
        return "Retreat"

    elif currentMode == "Charge Capsule":
      if captured:
        return "Travel to Center"
      elif (not canChargeCapsule) and defenderNear:
        return "Retreat"
      elif (not defenderNear) and shallowTargetsExist and (not holdingEnoughFood) and (not significantAdvantage):
        return "Shallow Offense"
      elif (not defenderNear) and (not shallowTargetsExist) and (not holdingEnoughFood) and (not significantAdvantage):
        return "Deep Offense"
      elif not isCapsuleOn:
        return "Charge Capsule"
      else:
        return "Shallow Offense"

    elif currentMode == "Intercept Invader":
      if captured:
        return "Travel to Center"
      elif invaderNear and (not onOffense) and ((otherMode != "Charge Invader" and otherMode != "Intercept Invader" and otherMode != "Sentry") or needHelp):
        return "Charge Invader"
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

    else: #Default
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
      enemyLocationPredictor.update(gameState, self, gameState.getAgentPosition(self.teamIndex), pos)
    investigatePoints = enemyLocationPredictor.getPositionsToInvestigate()
    if pos in investigatePoints:
      enemyLocationPredictor.removePositionFromInvestigation(pos)
    pastMode = self.mode
    
    
    self.mode = self.getNextMode(gameState, verbose = False)
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

    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
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
          tempFood = getShallowFood(gameState, self, self.red).asList()
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
          tempFood = getShallowFood(gameState, self, self.red).asList()
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
      self.dotsHeldByInvaders += len(self.prevFoodDef.asList()) - len(self.getFoodYouAreDefending(gameState).asList())
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

    #Priorities
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

  def getFeatures(self, gameState, mode, action):
    pass

  def getPriorities(self, mode):
    pass

  def claimTarget(self, target, investigate, targetId, pos, otherPos, override = False):
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
          enemyLocationPredictor.removePositionFromInvestigation(target)
        target1 = target
        ignoreCounter1 = 12
        return True
    else:
      if self.getMazeDistance(target, pos) < self.getMazeDistance(target, otherPos) or override:
        if investigate:
          enemyLocationPredictor.removePositionFromInvestigation(target)
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


class OffensiveReflexAgent(ReflexCaptureAgent):

  def getRisk(self, gameState, newPos):
    g = enemyLocationPredictor.getPositionPossibleGrid()
    walls = gameState.getWalls()
    risk = 0.0
    total = 0.0
    for r in range(walls.height):
      for c in range(walls.width):
        pos = (c,r)
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
    x,y = pos
    investigatePoints = enemyLocationPredictor.getPositionsToInvestigate()
    avoidPoints = enemyLocationPredictor.getPositionsToAvoid()

    isRed = self.red
    enemies = [successor.getAgentState(i) for i in self.getOpponents(gameState)]
    enemiesKnown = [a for a in enemies if a.getPosition() != None]
    invaders = [a for a in enemies if a.isPacman]
    invadersKnown = [a for a in enemies if a.isPacman and a.getPosition() != None]
    defenders = [a for a in enemies if not a.isPacman]
    defendersKnown = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    
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

    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
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
    features['capsuleDist'] = -1 * getMinDistToCapsule(successor, self, pos, self.getCapsules(successor))
    features['foodEaten'] = -1 * len(self.getFood(successor).asList())
    if len(deepFood) == 0:
      features['distToDeepFood'] = 0
    else:
      features['distToDeepFood'] = -1 * min([self.getMazeDistance(pos, food) for food in deepFood])
    if len(shallowFood) == 0:
      features['distToShallowFood'] = 0
    else:
      features['distToShallowFood'] = -1 * min([self.getMazeDistance(pos, food) for food in shallowFood])

    if self.target is None:
      features['teamDist'] = 0
    else:
      features['teamDist'] = self.getMazeDistance(pos, teamPos)

    features['enemyPredictOffense'] = -1 * self.getRisk(gameState, pos)
    features['enemyPredictDefense'] = -1 * features['enemyPredictOffense']

    

    if self.target is None:
      features['targetDist'] = 0
    else:
      features['targetDist'] = -1 * self.getMazeDistance(pos, self.target)
    

    if state.isPacman:
      features['onOffense'] = 1
      features['onDefense'] = 1
    else:
      features['onOffense'] = 0
      features['onDefense'] = 1

    if len(self.getCapsulesYouAreDefending(successor)) > 0:
      features['capsuleDist'] = -1 * sum([self.getMazeDistance(pos, capsule) for capsule in self.getCapsulesYouAreDefending(successor)])
    else:
      features['capsuleDist'] = 0

    if len(self.getCapsules(successor)) > 0:
      features['capsuleDistOffense'] = -1 * min([self.getMazeDistance(pos, capsule) for capsule in self.getCapsules(successor)])
    else:
      features['capsuleDistOffense'] = 0

    features['sentryScore'] = (features['capsuleDist'] * random1) + (features['distToSafety'] * (1-random1)) + features['teamDist']
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
