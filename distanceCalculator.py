# distanceCalculator.py
# ---------------------
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


"""
This file contains a Distancer object which computes and
caches the shortest path between any two points in the maze.
该文件包含一个 Distancer 对象，用于计算并缓存迷宫中任意两点之间的最短路径。

Example:
distancer = Distancer(gameState.data.layout)
distancer.getDistance( (1,1), (10,10) )
"""

import sys, time, random

class Distancer:
	def __init__(self, layout, default = 10000):
		"""
		Initialize with Distancer(layout).  Changing default is unnecessary.
		初始化，没有必要改变初始值
		"""
		self._distances = None
		self.default = default
		self.dc = DistanceCalculator(layout, self, default)

	def getMazeDistances(self):
		self.dc.run()

	def getDistance(self, pos1, pos2):
		"""
		The getDistance function is the only one you'll need after you create the object.
  		创建对象后，getDistance 函数是您唯一需要的函数。
		"""
		if self._distances == None:
			return manhattanDistance(pos1, pos2)
		if isInt(pos1) and isInt(pos2):
			return self.getDistanceOnGrid(pos1, pos2)
		pos1Grids = getGrids2D(pos1)
		pos2Grids = getGrids2D(pos2)
		bestDistance = self.default
		for pos1Snap, snap1Distance in pos1Grids:
			for pos2Snap, snap2Distance in pos2Grids:
				gridDistance = self.getDistanceOnGrid(pos1Snap, pos2Snap)
				distance = gridDistance + snap1Distance + snap2Distance
				if bestDistance > distance:
					bestDistance = distance
		return bestDistance

	def getDistanceOnGrid(self, pos1, pos2):
		key = (pos1, pos2)
		if key in self._distances:
			return self._distances[key]
		else:
			raise Exception("Positions not in grid: " + str(key))

	def isReadyForMazeDistance(self):
		return self._distances != None

def manhattanDistance(x, y ):
	return abs( x[0] - y[0] ) + abs( x[1] - y[1] )

def isInt(pos):
	x, y = pos
	return x == int(x) and y == int(y)

def getGrids2D(pos):
	grids = []
	for x, xDistance in getGrids1D(pos[0]):
		for y, yDistance in getGrids1D(pos[1]):
			grids.append(((x, y), xDistance + yDistance))
	return grids

def getGrids1D(x):
	intX = int(x)
	if x == int(x):
		return [(x, 0)]
	return [(intX, x-intX), (intX+1, intX+1-x)]

##########################################
# MACHINERY FOR COMPUTING MAZE DISTANCES #
##########################################

distanceMap = {}

class DistanceCalculator:
	def __init__(self, layout, distancer, default = 10000):
		self.layout = layout
		self.distancer = distancer
		self.default = default

	def run(self):
		global distanceMap

		if self.layout.walls not in distanceMap:
			distances = computeDistances(self.layout)
			distanceMap[self.layout.walls] = distances
		else:
			distances = distanceMap[self.layout.walls]

		self.distancer._distances = distances

def computeDistances(layout):
		"Runs UCS to all other positions from each position"
		"""
		利用广度优先搜索（BFS）算法计算其到其他位置的最短距离。
		具体而言，通过维护一个优先级队列（PriorityQueue）和一个距离字典（dist），
		在每次迭代中从队列中取出一个节点，
		并将其邻近的非墙壁位置加入队列并更新最短距离。
		最终将计算得到的最短距离存储在distances字典中，并返回该字典。
		"""
		distances = {}
		allNodes = layout.walls.asList(False)
		for source in allNodes:
				dist = {}	# 记录节点到起始节点的距离
				closed = {}	# 记录已经访问过的节点
				for node in allNodes:	
						dist[node] = sys.maxsize	# 初始距离设为正无穷
				import util
				queue = util.PriorityQueue()
				queue.push(source, 0)
				dist[source] = 0
				while not queue.isEmpty():
						node = queue.pop()
						if node in closed:
								continue
						closed[node] = True
						nodeDist = dist[node]
						adjacent = []
						x, y = node
						if not layout.isWall((x,y+1)):
								adjacent.append((x,y+1))
						if not layout.isWall((x,y-1)):
								adjacent.append((x,y-1) )
						if not layout.isWall((x+1,y)):
								adjacent.append((x+1,y) )
						if not layout.isWall((x-1,y)):
								adjacent.append((x-1,y))
						for other in adjacent:
								if not other in dist:
										continue
								oldDist = dist[other]
								newDist = nodeDist+1
								if newDist < oldDist:
										dist[other] = newDist
										queue.push(other, newDist)
				for target in allNodes:
						distances[(target, source)] = dist[target]
		return distances
"""
import sys
import heapq

def dijkstra(graph, start):
    # 初始化距离字典，将起始节点距离设为0，其余节点距离设为正无穷
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    # 使用堆来保存待处理节点，(距离, 节点) 的形式
    queue = [(0, start)

    while queue:
        # 从堆中取出距离最短的节点
        current_distance, current_node = heapq.heappop(queue)
        # 如果当前节点的距离大于已经记录的距离，则忽略
        if current_distance > distances[current_node]:
            continue
        # 遍历当前节点的相邻节点
        for neighbor, weight in graph[current_node].items():
            # 计算当前节点经过相邻节点到达目标节点的距禂
            distance = current_distance + weight
            # 如果经过相邻节点到达目标节点的距愈小于已经记录的距愈，则更新距愈并加入堆中
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))
    
    return distances

def computeDistancesWithDijkstra(layout):
    # 构建图，表示迷宫中各个节点之间的连接关系
    graph = {}
    allNodes = layout.walls.asList(False)
    for node in allNodes:
        graph[node] = {}
        x, y = node
        # 检查相邻节点是否是墙，并添加到图中
        if not layout.isWall((x, y + 1)):
            graph[node][(x, y + 1)] = 1
        if not layout.isWall((x, y - 1)):
            graph[node][(x, y - 1)] = 1
        if not layout.isWall((x + 1, y)):
            graph[node][(x + 1, y)] = 1
        if not layout.isWall((x - 1, y)):
            graph[node][(x - 1, y)] = 1

    distances = {}
    for node in allNodes:
        # 使用 Dijkstra 算法计算每个节点到其他节点之间的最短距离
        distances[node] = dijkstra(graph, node)

    return distances

"""

def getDistanceOnGrid(distances, pos1, pos2):
		key = (pos1, pos2)
		if key in distances:
			return distances[key]
		return 100000

