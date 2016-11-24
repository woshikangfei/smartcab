# Content: Reinforcement Learning
## Project: Train a Smartcab How to Drive

## Project Overview

In this project you will apply reinforcement learning techniques for a self-driving agent in a simplified world to aid it in effectively reaching its destinations in the allotted time. You will first investigate the environment the agent operates in by constructing a very basic driving implementation. Once your agent is successful at operating within the environment, you will then identify each possible state the agent can be in when considering such things as traffic lights and oncoming traffic at each intersection. With states identified, you will then implement a Q-Learning algorithm for the self-driving agent to guide the agent towards its destination within the allotted time. Finally, you will improve upon the Q-Learning algorithm to find the best configuration of learning and exploration factors to ensure the self-driving agent is reaching its destinations with consistently positive results.

## Description
In the not-so-distant future, taxicab companies across the United States no longer employ human drivers to operate their fleet of vehicles. Instead, the taxicabs are operated by self-driving agents, known as *smartcabs*, to transport people from one location to another within the cities those companies operate. In major metropolitan areas, such as Chicago, New York City, and San Francisco, an increasing number of people have come to depend on *smartcabs* to get to where they need to go as safely and reliably as possible. Although *smartcabs* have become the transport of choice, concerns have arose that a self-driving agent might not be as safe or reliable as human drivers, particularly when considering city traffic lights and other vehicles. To alleviate these concerns, your task as an employee for a national taxicab company is to use reinforcement learning techniques to construct a demonstration of a *smartcab* operating in real-time to prove that both safety and reliability can be achieved.
在不远的将来,出租车公司在美国不再使用人类的汽车司机来操作他们的舰队。相反,出租车是由自动驾驶人员,称为* smartcabs *,人们从一个地方运输到另一个城市内的这些公司运作。等主要大城市,芝加哥,纽约,旧金山,越来越多的人开始依赖* smartcabs *到他们需要去的地方尽可能安全、可靠。虽然* smartcabs *已经成为选择的运输,担忧起来,无人驾驶代理可能不会像人类一样安全或可靠的司机,尤其是当考虑城市交通信号灯和其他车辆。缓解这些担忧,你的任务作为一个国家出租车公司的员工是利用强化学习技术来构建一个实时演示* smartcab *操作证明都可以实现安全性和可靠性。

## Software Requirements
This project uses the following software and Python libraries:

- [Python 2.7](https://www.python.org/download/releases/2.7/)
- [NumPy](http://www.numpy.org/)
- [pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [PyGame](http://pygame.org/)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. Make sure that you select the Python 2.7 installer and not the Python 3.x installer. `pygame` can then be installed using one of the following commands:

Mac:  `conda install -c https://conda.anaconda.org/quasiben pygame`  
Windows: `conda install -c https://conda.anaconda.org/tlatorre pygame`  
Linux:  `conda install -c https://conda.anaconda.org/prkrekel pygame`  

## Fixing Common PyGame Problems

The PyGame library can in some cases require a bit of troubleshooting to work correctly for this project. While the PyGame aspect of the project is not required for a successful submission  (you can complete the project without a visual simulation, although it is more difficult), it is very helpful to have it working! If you encounter an issue with PyGame, first see these helpful links below that are developed by communities of users working with the library:
- [Getting Started](https://www.pygame.org/wiki/GettingStarted)
- [PyGame Information](http://www.pygame.org/wiki/info)
- [Google Group](https://groups.google.com/forum/#!forum/pygame-mirror-on-google-groups)
- [PyGame subreddit](https://www.reddit.com/r/pygame/)

### Problems most often reported by students
_"PyGame won't install on my machine; there was an issue with the installation."_  
**Solution:** As has been recommended for previous projects, Udacity suggests that you are using the Anaconda distribution of Python, which can then allow you to install PyGame through the `conda`-specific command.

_"I'm seeing a black screen when running the code; output says that it can't load car images."_  
**Solution:** The code will not operate correctly unless it is run from the top-level directory for `smartcab`. The top-level directory is the one that contains the **README** and the project notebook.

If you continue to have problems with the project code in regards to PyGame, you can also [use the discussion forums](https://discussions.udacity.com/c/nd009-reinforcement-learning) to find posts from students that encountered issues that you may be experiencing. Additionally, you can seek help from a swath of students in the [MLND Student Slack Community](http://mlnd.slack.com).

## Starting the Project

For this assignment, you can find the `smartcab` folder containing the necessary project files on the [Machine Learning projects GitHub](https://github.com/udacity/machine-learning), under the `projects` folder. You may download all of the files for projects we'll use in this Nanodegree program directly from this repo. Please make sure that you use the most recent version of project files when completing a project!

This project contains three directories:

- `/logs/`: This folder will contain all log files that are given from the simulation when specific prerequisites are met.这个文件夹将包含所有日志文件时从模拟给出的特定先决条件得到满足。
- `/images/`: This folder contains various images of cars to be used in the graphical user interface. You will not need to modify or create any files in this directory.这个文件夹包含各种图像的汽车中使用的图形用户界面。你不需要修改或创建任何文件在这个目录中。
- `/smartcab/`: This folder contains the Python scripts that create the environment, graphical user interface, the simulation, and the agents. You will not need to modify or create any files in this directory except for `agent.py`.这个文件夹包含Python脚本,创建环境,图形用户界面,模拟和代理。你不需要修改或创建任何文件在这个目录中除了“agent.py”。

It also contains two files:
- `smartcab.ipynb`: This is the main file where you will answer questions and provide an analysis for your work.这是主文件,你将回答问题对你的工作和提供了一个分析。
-`visuals.py`: This Python script provides supplementary visualizations for the analysis. Do not modify.这个Python脚本提供了补充分析的可视化。不要修改。

Finally, in `/smartcab/` are the following four files:
- **Modify:**
  - `agent.py`: This is the main Python file where you will be performing your work on the project.这是主Python文件,您将执行你的工作在这个项目。
- **Do not modify:**
  - `environment.py`: This Python file will create the *smartcab* environment.这个Python文件将创建* smartcab *环境。
  - `planner.py`: This Python file creates a high-level planner for the agent to follow towards a set goal.这个Python文件创建一个高级规划师为代理对一组目标。
  - `simulation.py`: This Python file creates the simulation and graphical user interface. 这个Python文件创建仿真和图形用户界面

### Running the Code
In a terminal or command window, navigate to the top-level project directory `smartcab/` (that contains the two project directories) and run one of the following commands:
在一个终端或命令窗口中,导航到顶级项目目录“smartcab /”(包含两个项目目录)并运行以下命令之一:
`python smartcab/agent.py` or  
`python -m smartcab.agent`

This will run the `agent.py` file and execute your implemented agent code into the environment. Additionally, use the command `jupyter notebook smartcab.ipynb` from this same directory to open up a browser window or tab to work with your analysis notebook. Alternatively, you can use the command `jupyter notebook` or `ipython notebook` and navigate to the notebook file in the browser window that opens. Follow the instructions in the notebook and answer each question presented to successfully complete the implementation necessary for your `agent.py` agent file. A **README** file has also been provided with the project files which may contain additional necessary information or instruction for the project.
这将运行的代理。py文件并执行你的代理代码到环境中实现的。此外,使用命令“jupyter笔记本smartcab。ipynb从这个目录来打开一个浏览器窗口或选项卡与你分析笔记本。或者,您可以使用命令“jupyter笔记本”或“ipython笔记本”并导航到笔记本文件浏览器窗口中打开。按照说明在笔记本和回答每个问题提交成功完成所需的实现你的代理。py的代理文件。* * * * README文件也提供了必要的项目文件可能包含额外的信息或指令的项目。
## Definitions

### Environment
The *smartcab* operates in an ideal, grid-like city (similar to New York City), with roads going in the North-South and East-West directions. Other vehicles will certainly be present on the road, but there will be no pedestrians to be concerned with. At each intersection there is a traffic light that either allows traffic in the North-South direction or the East-West direction. U.S. Right-of-Way rules apply: 
* smartcab *操作在一个理想的、网状城市(类似于纽约),南北和东西方向与道路。其他车辆肯定会出现在路上,但不会有行人关心。在每一个十字路口有红绿灯,要么允许南北方向和东西方向的交通。美国申请优先权规则:
- On a green light, a left turn is permitted if there is no oncoming traffic making a right turn or coming straight through the intersection.绿灯,左转是允许的,如果没有车流进行或直接通过十字路口向右拐。
- On a red light, a right turn is permitted if no oncoming traffic is approaching from your left through the intersection.允许右转红灯,如果没有从迎面而来的车辆左穿过十字路口。
To understand how to correctly yield to oncoming traffic when turning left, you may refer to [this official drivers? education video](https://www.youtube.com/watch?v=TW0Eq2Q-9Ac), or [this passionate exposition](https://www.youtube.com/watch?v=0EdkxI6NeuA).
了解如何正确地屈服于车流左转的时候,你可以引用这个官方的司机吗?教育视频)(https://www.youtube.com/watch?v=TW0Eq2Q-9Ac),或(这充满激情的博览会)(https://www.youtube.com/watch?v=0EdkxI6NeuA)。

### Inputs and Outputs
Assume that the *smartcab* is assigned a route plan based on the passengers? starting location and destination. The route is split at each intersection into waypoints, and you may assume that the *smartcab*, at any instant, is at some intersection in the world. Therefore, the next waypoint to the destination, assuming the destination has not already been reached, is one intersection away in one direction (North, South, East, or West). The *smartcab* has only an egocentric view of the intersection it is at: It can determine the state of the traffic light for its direction of movement, and whether there is a vehicle at the intersection for each of the oncoming directions. For each action, the *smartcab* may either idle at the intersection, or drive to the next intersection to the left, right, or ahead of it. Finally, each trip has a time to reach the destination which decreases for each action taken (the passengers want to get there quickly).  If the allotted time becomes zero before reaching the destination, the trip has failed.
假设* smartcab *分配路线计划基于乘客吗?起始位置和目的地。路线在每个路口分为锚点,你可能认为* smartcab *,在任何时候,在一些世界上的十字路口。因此,下一个路标的目的地,如果目的还没有达到,是一个十字路口在一个方向(北、南、东、西)。* smartcab *只有一个自我中心的十字路口:它可以确定交通灯的状态的方向运动,和是否有车辆在十字路口为每个迎面而来的方向。对于每个操作,* smartcab *可以空闲的十字路口,或开车到下一个路口向左,右,或提前。最后,每个旅行有一个时间到达目的地,减少为每个行动(乘客想快速到达那里)。如果规定时间为零在到达目的地之前,这次旅行已经失败了。
### Rewards and Goal
The *smartcab* will receive positive or negative rewards based on the action it as taken. Expectedly, the *smartcab* will receive a small positive reward when making a good action, and a varying amount of negative reward dependent on the severity of the traffic violation it would have committed. Based on the rewards and penalties the *smartcab* receives, the self-driving agent implementation should learn an optimal policy for driving on the city roads while obeying traffic rules, avoiding accidents, and reaching passengers? destinations in the allotted time.
* smartcab *将得到积极或消极的奖励根据它采取行动。最少、* smartcab *将收到一个小积极奖励时一个很好的行动,和不同数量的负回报依赖于交通违章的严重性会提交。基于* smartcab *收到奖励和处罚,无人驾驶代理实现应该学习最优政策在城市道路驾驶,同时遵守交通规则,避免事故的发生,达到乘客呢?在规定时间内目的地。

## Submitting the Project

### Evaluation
Your project will be reviewed by a Udacity reviewer against the **<a href="https://review.udacity.com/#!/rubrics/106/view" target="_blank">Train a Smartcab to Drive project rubric</a>**. Be sure to review this rubric thoroughly and self-evaluate your project before submission. All criteria found in the rubric must be *meeting specifications* for you to pass.

### Submission Files
When you are ready to submit your project, collect the following files and compress them into a single archive for upload. Alternatively, you may supply the following files on your GitHub Repo in a folder named `smartcab` for ease of access:
当你准备提交你的项目,收集以下文件并压缩成一个档案上传。或者,你在GitHub回购可能供应下列文件在一个文件夹命名为“smartcab”易于访问:
- The `agent.py` Python file with all code implemented as required in the instructed tasks.
- The `/logs/` folder which should contain **five** log files that were produced from your simulation and used in the analysis.
- The `smartcab.ipynb` notebook file with all questions answered and all visualization cells executed and displaying results.
 - An **HTML** export of the project notebook with the name **report.html**. This file *must* be present for your project to be evaluated.

Once you have collected these files and reviewed the project rubric, proceed to the project submission page.
一旦你收集了这些文件,并回顾了项目标题,继续该项目提交页面。
### I'm Ready!
When you're ready to submit your project, click on the **Submit Project** button at the bottom of the page.
当你准备提交你的项目,点击* *提交项目* *按钮在页面的底部。
If you are having any problems submitting your project or wish to check on the status of your submission, please email us at **machine-support@udacity.com** or visit us in the <a href="http://discussions.udacity.com" target="_blank">discussion forums</a>.
如果你有任何问题提交你的项目或希望检查提交的状态,请电子邮件我们在machine-support@udacity.com * * * *或访问我们的< a href = " http://discussions.udacity.com " target = "平等" >论坛< / >。
### What's Next?
You will get an email as soon as your reviewer has feedback for you. In the meantime, review your next project and feel free to get started on it or the courses supporting it!