## 一.已解决问题
### 1.  myteam.py到底要写些什么?   
    1. 一个函数对位接口creatTeam()，感觉可以照搬myteam.py里的??
    2. 几个Agent类,这里个人认为可以组员各自写一个Agent类,选比较好的那个,或者里面的方法取更好的那个,最后汇总构成最终的Agent类
### 2. Agent类参考哪些代码?    
    1. myteam.py里的DummmyAgent类(就是个假的,个人认为没什么参考价值,不过注释有必要看下)
    2.  !!! 所有Agent类的算法原型是CaptureAgent -- 位于 captureAgents.py 文件!!!
        建议编写算法时继承 CaptureAgent 类
    3. baseline.py里的ReflexCaptureAgent类
    攻击型Agent OffensiveReflexAgent 与防御型AgentDefensiveReflexAgent
    （后两者继承的是ReflexCaptureAgent类） 
    4. 其他资料??

## 二.待解决问题
1. myteam.py中方法registerInitialState()的注释里所说的"使用曼哈顿距离而不是迷宫距离以节省初始化时间"是不是可以写一写? 
2. 什么是噪声距离?
3. capture.py里方法halfList(l, grid, red)中传入的l是什么?
4. CaptureAgent.py中提到的factory是什么?
5. 需要写的Agent类里有一个必要的方法chooseAction(),
    在CaptureAgent.py中的captureAgent类中的方法chooseAction()里有提到
     "重写此方法可制作一个好的agent。它应在时限内返回合法action",
     这里的action是什么,仅仅只是智能体上下左右移动吗?
