# Intersection-area-fraction Error for Path Following of Wing-in-ground Crafts based on Deep Reinforcement Learning
 
* **Description on directories:**  The codes are presented in seperate dierectory according to the section of 'Cases and analysis' in the manuscript. Since two environments are used to examine the performance of the conventional cross-track error and intersection-area-fraction error, there are two main directories to represent each environment. As for each sub case in the environment of piece-wise lines, eight different cases are at sub-sub directory. For each case, trajectories trained by cross-track error are collected, so there is one seed. While for the situation on intersection-area-fraction error, there are 5 seeds for the training to pick the one rewarded mostly as the result. So, there are 5 dierectories at each sub-sub directory in the 'intersection-area-fraction error'. The same is true for the second environment。

* **Prerequisites for running codes:**  Pytorch is needed to be installed in advance. In order to run the code at each sub-sub directory, the file 'datacollection.npy' should be downloaded and then placed in the root directory, meaning that it is placed with 'environment of piece-wise lines', etc. Since the website of Github limits the size of files, the necessary file and main results are placed in the same directory and they are upload in onedrive. The link is https://maildluteducn-my.sharepoint.com/:f:/g/personal/huhuan2019_mail_dlut_edu_cn/EhDPV3MMc9hMjymhPDYc5-4BtJVbdRgBjAXtArgBroaluA?e=gSb9hL. The file 'datacollection.npy' is used for the calculation of aerodynamics for the WIG.


* **Main code file:** At the each sub-sub directory, there are six files for the training, which are as following:
1. *main.py*. It's main function and the training environment and reward function are defined.

2. *run.py*. It's algorithm of training.

3. *replay.py*. It's classes of replay buffer during training.

4. *net.py*. It's classes of neural network for the algorithm of Proximal Policy Optimization (PPO).

5. *agent.py*. It's classes of agents for PPO.

6. *aeropy_fix.py*. It's functions for aerodynamics and kinectics of the wing-in-ground craft (WIG) during training.


