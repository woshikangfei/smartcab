# Content: Reinforcement Learning
## Project: Train a Smartcab How to Drive

## Project Overview

In this project you will apply reinforcement learning techniques for a self-driving agent in a simplified world to aid it in effectively reaching its destinations in the allotted time. You will first investigate the environment the agent operates in by constructing a very basic driving implementation. Once your agent is successful at operating within the environment, you will then identify each possible state the agent can be in when considering such things as traffic lights and oncoming traffic at each intersection. With states identified, you will then implement a Q-Learning algorithm for the self-driving agent to guide the agent towards its destination within the allotted time. Finally, you will improve upon the Q-Learning algorithm to find the best configuration of learning and exploration factors to ensure the self-driving agent is reaching its destinations with consistently positive results.

## Description
In the not-so-distant future, taxicab companies across the United States no longer employ human drivers to operate their fleet of vehicles. Instead, the taxicabs are operated by self-driving agents, known as *smartcabs*, to transport people from one location to another within the cities those companies operate. In major metropolitan areas, such as Chicago, New York City, and San Francisco, an increasing number of people have come to depend on *smartcabs* to get to where they need to go as safely and reliably as possible. Although *smartcabs* have become the transport of choice, concerns have arose that a self-driving agent might not be as safe or reliable as human drivers, particularly when considering city traffic lights and other vehicles. To alleviate these concerns, your task as an employee for a national taxicab company is to use reinforcement learning techniques to construct a demonstration of a *smartcab* operating in real-time to prove that both safety and reliability can be achieved.
�ڲ�Զ�Ľ���,���⳵��˾����������ʹ�����������˾�����������ǵĽ��ӡ��෴,���⳵�����Զ���ʻ��Ա,��Ϊ* smartcabs *,���Ǵ�һ���ط����䵽��һ�������ڵ���Щ��˾����������Ҫ�����,֥�Ӹ�,ŦԼ,�ɽ�ɽ,Խ��Խ����˿�ʼ����* smartcabs *��������Ҫȥ�ĵط������ܰ�ȫ���ɿ�����Ȼ* smartcabs *�Ѿ���Ϊѡ�������,��������,���˼�ʻ������ܲ���������һ����ȫ��ɿ���˾��,�����ǵ����ǳ��н�ͨ�źŵƺ�����������������Щ����,���������Ϊһ�����ҳ��⳵��˾��Ա��������ǿ��ѧϰ����������һ��ʵʱ��ʾ* smartcab *����֤��������ʵ�ְ�ȫ�ԺͿɿ��ԡ�

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

- `/logs/`: This folder will contain all log files that are given from the simulation when specific prerequisites are met.����ļ��н�����������־�ļ�ʱ��ģ��������ض��Ⱦ������õ����㡣
- `/images/`: This folder contains various images of cars to be used in the graphical user interface. You will not need to modify or create any files in this directory.����ļ��а�������ͼ���������ʹ�õ�ͼ���û����档�㲻��Ҫ�޸Ļ򴴽��κ��ļ������Ŀ¼�С�
- `/smartcab/`: This folder contains the Python scripts that create the environment, graphical user interface, the simulation, and the agents. You will not need to modify or create any files in this directory except for `agent.py`.����ļ��а���Python�ű�,��������,ͼ���û�����,ģ��ʹ����㲻��Ҫ�޸Ļ򴴽��κ��ļ������Ŀ¼�г��ˡ�agent.py����

It also contains two files:
- `smartcab.ipynb`: This is the main file where you will answer questions and provide an analysis for your work.�������ļ�,�㽫�ش��������Ĺ������ṩ��һ��������
-`visuals.py`: This Python script provides supplementary visualizations for the analysis. Do not modify.���Python�ű��ṩ�˲�������Ŀ��ӻ�����Ҫ�޸ġ�

Finally, in `/smartcab/` are the following four files:
- **Modify:**
  - `agent.py`: This is the main Python file where you will be performing your work on the project.������Python�ļ�,����ִ����Ĺ����������Ŀ��
- **Do not modify:**
  - `environment.py`: This Python file will create the *smartcab* environment.���Python�ļ�������* smartcab *������
  - `planner.py`: This Python file creates a high-level planner for the agent to follow towards a set goal.���Python�ļ�����һ���߼��滮ʦΪ�����һ��Ŀ�ꡣ
  - `simulation.py`: This Python file creates the simulation and graphical user interface. ���Python�ļ����������ͼ���û�����

### Running the Code
In a terminal or command window, navigate to the top-level project directory `smartcab/` (that contains the two project directories) and run one of the following commands:
��һ���ն˻��������,������������ĿĿ¼��smartcab /��(����������ĿĿ¼)��������������֮һ:
`python smartcab/agent.py` or  
`python -m smartcab.agent`

This will run the `agent.py` file and execute your implemented agent code into the environment. Additionally, use the command `jupyter notebook smartcab.ipynb` from this same directory to open up a browser window or tab to work with your analysis notebook. Alternatively, you can use the command `jupyter notebook` or `ipython notebook` and navigate to the notebook file in the browser window that opens. Follow the instructions in the notebook and answer each question presented to successfully complete the implementation necessary for your `agent.py` agent file. A **README** file has also been provided with the project files which may contain additional necessary information or instruction for the project.
�⽫���еĴ���py�ļ���ִ����Ĵ�����뵽������ʵ�ֵġ�����,ʹ�����jupyter�ʼǱ�smartcab��ipynb�����Ŀ¼����һ����������ڻ�ѡ���������ʼǱ�������,������ʹ�����jupyter�ʼǱ�����ipython�ʼǱ������������ʼǱ��ļ�����������д򿪡�����˵���ڱʼǱ��ͻش�ÿ�������ύ�ɹ���������ʵ����Ĵ���py�Ĵ����ļ���* * * * README�ļ�Ҳ�ṩ�˱�Ҫ����Ŀ�ļ����ܰ����������Ϣ��ָ�����Ŀ��
## Definitions

### Environment
The *smartcab* operates in an ideal, grid-like city (similar to New York City), with roads going in the North-South and East-West directions. Other vehicles will certainly be present on the road, but there will be no pedestrians to be concerned with. At each intersection there is a traffic light that either allows traffic in the North-South direction or the East-West direction. U.S. Right-of-Way rules apply: 
* smartcab *������һ������ġ���״����(������ŦԼ),�ϱ��Ͷ����������·�����������϶��������·��,�����������˹��ġ���ÿһ��ʮ��·���к��̵�,Ҫô�����ϱ�����Ͷ�������Ľ�ͨ��������������Ȩ����:
- On a green light, a left turn is permitted if there is no oncoming traffic making a right turn or coming straight through the intersection.�̵�,��ת�������,���û�г������л�ֱ��ͨ��ʮ��·�����ҹա�
- On a red light, a right turn is permitted if no oncoming traffic is approaching from your left through the intersection.������ת���,���û�д�ӭ������ĳ����󴩹�ʮ��·�ڡ�
To understand how to correctly yield to oncoming traffic when turning left, you may refer to [this official drivers? education video](https://www.youtube.com/watch?v=TW0Eq2Q-9Ac), or [this passionate exposition](https://www.youtube.com/watch?v=0EdkxI6NeuA).
�˽������ȷ�������ڳ�����ת��ʱ��,�������������ٷ���˾����?������Ƶ)(https://www.youtube.com/watch?v=TW0Eq2Q-9Ac),��(���������Ĳ�����)(https://www.youtube.com/watch?v=0EdkxI6NeuA)��

### Inputs and Outputs
Assume that the *smartcab* is assigned a route plan based on the passengers? starting location and destination. The route is split at each intersection into waypoints, and you may assume that the *smartcab*, at any instant, is at some intersection in the world. Therefore, the next waypoint to the destination, assuming the destination has not already been reached, is one intersection away in one direction (North, South, East, or West). The *smartcab* has only an egocentric view of the intersection it is at: It can determine the state of the traffic light for its direction of movement, and whether there is a vehicle at the intersection for each of the oncoming directions. For each action, the *smartcab* may either idle at the intersection, or drive to the next intersection to the left, right, or ahead of it. Finally, each trip has a time to reach the destination which decreases for each action taken (the passengers want to get there quickly).  If the allotted time becomes zero before reaching the destination, the trip has failed.
����* smartcab *����·�߼ƻ����ڳ˿���?��ʼλ�ú�Ŀ�ĵء�·����ÿ��·�ڷ�Ϊê��,�������Ϊ* smartcab *,���κ�ʱ��,��һЩ�����ϵ�ʮ��·�ڡ����,��һ��·���Ŀ�ĵ�,���Ŀ�Ļ�û�дﵽ,��һ��ʮ��·����һ������(�����ϡ�������)��* smartcab *ֻ��һ���������ĵ�ʮ��·��:������ȷ����ͨ�Ƶ�״̬�ķ����˶�,���Ƿ��г�����ʮ��·��Ϊÿ��ӭ������ķ��򡣶���ÿ������,* smartcab *���Կ��е�ʮ��·��,�򿪳�����һ��·������,��,����ǰ�����,ÿ��������һ��ʱ�䵽��Ŀ�ĵ�,����Ϊÿ���ж�(�˿�����ٵ�������)������涨ʱ��Ϊ���ڵ���Ŀ�ĵ�֮ǰ,��������Ѿ�ʧ���ˡ�
### Rewards and Goal
The *smartcab* will receive positive or negative rewards based on the action it as taken. Expectedly, the *smartcab* will receive a small positive reward when making a good action, and a varying amount of negative reward dependent on the severity of the traffic violation it would have committed. Based on the rewards and penalties the *smartcab* receives, the self-driving agent implementation should learn an optimal policy for driving on the city roads while obeying traffic rules, avoiding accidents, and reaching passengers? destinations in the allotted time.
* smartcab *���õ������������Ľ�����������ȡ�ж������١�* smartcab *���յ�һ��С��������ʱһ���ܺõ��ж�,�Ͳ�ͬ�����ĸ��ر������ڽ�ͨΥ�µ������Ի��ύ������* smartcab *�յ������ʹ���,���˼�ʻ����ʵ��Ӧ��ѧϰ���������ڳ��е�·��ʻ,ͬʱ���ؽ�ͨ����,�����¹ʵķ���,�ﵽ�˿���?�ڹ涨ʱ����Ŀ�ĵء�

## Submitting the Project

### Evaluation
Your project will be reviewed by a Udacity reviewer against the **<a href="https://review.udacity.com/#!/rubrics/106/view" target="_blank">Train a Smartcab to Drive project rubric</a>**. Be sure to review this rubric thoroughly and self-evaluate your project before submission. All criteria found in the rubric must be *meeting specifications* for you to pass.

### Submission Files
When you are ready to submit your project, collect the following files and compress them into a single archive for upload. Alternatively, you may supply the following files on your GitHub Repo in a folder named `smartcab` for ease of access:
����׼���ύ�����Ŀ,�ռ������ļ���ѹ����һ�������ϴ�������,����GitHub�ع����ܹ�Ӧ�����ļ���һ���ļ�������Ϊ��smartcab�����ڷ���:
- The `agent.py` Python file with all code implemented as required in the instructed tasks.
- The `/logs/` folder which should contain **five** log files that were produced from your simulation and used in the analysis.
- The `smartcab.ipynb` notebook file with all questions answered and all visualization cells executed and displaying results.
 - An **HTML** export of the project notebook with the name **report.html**. This file *must* be present for your project to be evaluated.

Once you have collected these files and reviewed the project rubric, proceed to the project submission page.
һ�����ռ�����Щ�ļ�,���ع�����Ŀ����,��������Ŀ�ύҳ�档
### I'm Ready!
When you're ready to submit your project, click on the **Submit Project** button at the bottom of the page.
����׼���ύ�����Ŀ,���* *�ύ��Ŀ* *��ť��ҳ��ĵײ���
If you are having any problems submitting your project or wish to check on the status of your submission, please email us at **machine-support@udacity.com** or visit us in the <a href="http://discussions.udacity.com" target="_blank">discussion forums</a>.
��������κ������ύ�����Ŀ��ϣ������ύ��״̬,������ʼ�������machine-support@udacity.com * * * *��������ǵ�< a href = " http://discussions.udacity.com " target = "ƽ��" >��̳< / >��
### What's Next?
You will get an email as soon as your reviewer has feedback for you. In the meantime, review your next project and feel free to get started on it or the courses supporting it!