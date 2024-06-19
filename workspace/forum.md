## 一.注意事项!!!
1.为了方便调试可以把capture.py中大约872行的有关'-r'和'-b'的defaut参数改成自己写的agent文件,运行文件前看看这个有没有修改回来

## 二.已解决问题
### 1.  myteam.py到底要写些什么?   
    1. 一个函数对外接口creatTeam()，感觉可以照搬myteam.py里的??
    2. 几个Agent类,这里个人认为可以组员各自写一个Agent类,选比较好的那个,或者里面的方法取更好的那个,最后汇总构成最终的Agent类
### 2. Agent类参考哪些代码?    
    1. myteam.py里的DummmyAgent类(就是个假的,个人认为没什么参考价值,不过注释有必要看下)
    2.  !!! 所有Agent类的算法原型是CaptureAgent -- 位于 captureAgents.py 文件!!!
        建议编写算法时继承 CaptureAgent 类
    3. baseline.py里的ReflexCaptureAgent类
    攻击型Agent OffensiveReflexAgent 与防御型AgentDefensiveReflexAgent
    （后两者继承的是ReflexCaptureAgent类） 
    4. 其他资料??
### 3. CaptureAgent.py中提到的factory是什么?  (6/18 22:00PM)
    应该是初始化时会调用的,没啥用
### 4. 需要写的Agent类里有一个必要的方法chooseAction(),action怎么得到 (6/18 22:00PM)
    可以参考baseline,py中的if语句部分,这一段的意思是:直到剩余可吃的豆子小于2时,选择回到起始点最短路径的action进行返回,否则返回的是能获取到豆子的最优action(这里使用方法evaluate()与max()获得的)

### 5.方法registerInitialState()的注释里所说的"使用曼哈顿距离而不是迷宫距离以节省初始化时间"是不是可以写一写?    (6/18 22:00PM)
    1.有点难度,最好不要改
    2.注释掉会影响captureagent.py中getMazeDistance()的使用

## 三.待解决问题

2. 什么是噪声距离?
3. capture.py里方法halfList(l, grid, red)中传入的l是什么?



