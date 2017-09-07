# Alpha Driver 
An end to end driving agent


## 1. Workflow
1. 开始工作前，从远程更新本地仓库，
```bash
git pull
```
或者直接用pycharm 的 
```bash
VCS -> update project
```
2. 开始工作，期间，可以自己决定是否 commit 或者 push 到远程仓库自己的分支
3. 工作直至完成一个测试通过的模块，决定合并到 master 分支
```bash
1. 进入 github PR页面https://github.com/Lukeeeeee/AlphaDriver/pulls
2. New pull request
3. 完成comment 等信息, 然后等待被merge
```

## 2. 代码结构的一些思路

1. 模型model 和功能 utils 尽可能解耦
2. 接口设计尽可能抽象

## 3. Torcs 游戏
[官方论文参考](https://arxiv.org/pdf/1304.1672.pdf)


## 4. TO DO LIST
0. LSMS + DDPG: /src/model/ddpg, actor. critic: 即基于 RNN 的 DDPG 模型
1. Environment： /src/environment/ 主要包括：
```bash
    1. 和 gym_torcs 的接口 /src/environment/torcsEnvironment
```
2. 对无预训练的 DDPG LSTM 的模型进行测试
3. preTrainer: /src/model/preTrainer，即模型预训练算法那的抽象模型
4. 实现具体的预训练算法，和准备训练数据
5. 完成带预训练的 DPPG LSTM 模型。并进行测试
6. 文档，继续维护