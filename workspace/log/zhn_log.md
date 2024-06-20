1. 给代码加了点注释,梳理了大致方向  24/06/11  11:00AM
2. 计划写一个Agent
3. python capture.py -r algo/zhn_agent -b baselineTeam 用于我写的agent与baselineTeam比赛  24/06/11  11:00AM

evaluate这里可能要下功夫

4.在ZHN_agent.py文件里写了个基类ApproximateAdversarialAgent,
5.动作选取通过alpha-beta 剪枝,看不见的敌人用贝叶斯推理近似
6.写攻击防御的agent只需要改evaluatestate函数给返回各个动作的值
7.在基类agent里写了几个辅助函数
    (1)getAgentPosition 除了原来的功能外,还可以猜测敌人的大概位置
    (2)agentIsPacman(self, agent, gameState)判断agent是否是pacman
    (3)getOpponentDistances(self, gameState)返回相对于此agent的对手的 ID 和距离
    (4)getFoodCarried(self, gameState) 获取圣商的food数量