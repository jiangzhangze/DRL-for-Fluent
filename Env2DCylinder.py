import numpy as np
import pandas as pd
import os
import os.path as osp
import random
import ansys.fluent.core as pyfluent
from decimal import Decimal
import gym
from gym import Env
from gym import spaces
import time



class RingBuffer():
    "A 1D ring buffer using numpy arrays"
    def __init__(self, length):
        self.data = np.zeros(length, dtype='f')
        self.index = 0

    def extend(self, x):
        "adds array x to ring buffer"
        x_index = (self.index + np.arange(x.size)) % self.data.size
        self.data[x_index] = x
        self.index = x_index[-1] + 1

    def get(self):
        "Returns the first-in-first-out data in the ring buffer"
        idx = (self.index + np.arange(self.data.size)) % self.data.size
        return self.data[idx]



project_dir = r'F:\MPC_2'



class Env2DCylinder(Env):
    """Environment for 2D flow simulation around a cylinder."""

    def __init__(self, size_history=80, number_steps_execution=50):
        # super(self, Env2DCylinder).__init__()
        self.min_jet = 0                                                       # 最低射流速度
        self.max_jet = 0.164                                                   # 最高射流速度
        self.episode_number = 0                                                # episode计数
        self.size_history = size_history                                       # history_parameters的大小，换句话说，这个是我们要读取多少cd、cl来计算reward
        self.number_steps_execution = number_steps_execution                   # 执行的步数。也就是说一个action执行的步数
        self.action_space = spaces.Box(low=self.min_jet, high=self.max_jet, shape=(1,), dtype=float)
        # 定义连续状态空间
        low = 0.0   # 状态空间的最小值
        high = 100000  # 状态空间的最大值
        shape = (9858,)  # 状态向量的形状
        self.observation_space = spaces.Box(low=low, high=high, shape=shape, dtype=float)
        self.episode_drags = np.array([])
        self.episode_lifts = np.array([])
        self.start_case = 'F:\\MPC_2\\case\\0.0000.case.h5.dat'
        self.max_step = 80
        self.jet_vec = 0
        self.start_class(complete_reset=True)

    def start_class(self,complete_reset=True):
        if complete_reset == False:
            self.solver_step = 0
        else:
            self.solver_step = 0
            self.flow_time = 491.23
            self.accumulated_drag = 0  # 这个是为了看整个episode的平均阻力
            self.accumulated_lift = 0  # 这个是为了看整个episode的平均升力

            self.history_parameters = {}  # 这个是为了记载history_parameters
            self.history_parameters["drag"] = RingBuffer(self.size_history)
            self.history_parameters["lift"] = RingBuffer(self.size_history)
            self.history_parameters["observation"] = RingBuffer(self.size_history)
            # 打开pyfluent
            self.session = pyfluent.launch_fluent(version='2d', precision='double', start_timeout=18000, processor_count=1)
            self.session.tui.define.user_defined.compiled_functions("load", "libudf")
            self.session.tui.define.user_defined.compiled_functions("compile","libudf","yes","inlet.c","","")
            self.session.tui.file.read_case_data(self.start_case)
            self.ready_to_use = True
            self.jet_vec = 0

        # 新环境命名episode，flowfield，cdcl
        self.episode_dir = osp.join(project_dir,r"data",r'episode_' + str(self.episode_number))
        self.flowfield_dir = osp.join(self.episode_dir, r"flowfield")
        self.cdcl_path = osp.join(self.episode_dir, r"cdcl.csv")
        self.flowfield_name = osp.join(self.flowfield_dir, r"episode_" + str(self.solver_step) + ".csv")

        self.session.tui.solve.report_files.edit("cdcl","file-name",self.cdcl_path)

        self.new_observation = self.get_obs()


    def step(self,action):
        # 新执行了一次迭代
        TimeExponent = 200*self.flow_time -1
        Name_Expressions = '''"(1-0.9^(200*(flow_time/1[s]) - {kkk}))*{action[0]} [kg/s] + 0.9^(200*flow_time/1[s] -{kkk})*{jec_vec} [kg/s]"'''.format(
            jec_vec=self.jet_vec, kkk=TimeExponent, action=action)                                                                                            #射流速度的Named_expressions表达式，（1-0.9^((t-491.23)/0.005+1))*action +0.9^((t-491.23)/0.005+1)*jec_vec
        self.session.tui.define.boundary_conditions.mass_flow_inlet("jet1", "yes", "yes", "no", Name_Expressions, "no", "0", "no", "yes")
        self.session.tui.define.boundary_conditions.mass_flow_outlet("jet2", "yes", "yes", "no", Name_Expressions, "no", "0", "no", "yes")
        self.session.tui.solve.dual_time_iterate("50", "10")
        self.jet_vec = action[0]





            # 获得流场速度值，获得阻力系数，升力系数
        self.flowfield_name = osp.join(self.flowfield_dir, r"episode_" + str(self.solver_step) + ".csv")
        self.new_observation = self.get_obs()[:,0]
        self.drag, self.lift = self.get_drag_lift()

            # 写进累计阻力，累计升力，和历史参数中
        self.write_history_parameters()
        self.accumulated_drag += self.drag
        self.accumulated_lift += self.lift

        self.solver_step += 1
        self.flow_time += 0.25
        next_state = np.reshape(self.new_observation, (1, 9858))

        reward = self.compute_reward()

        terminal = False
        if self.solver_step > (self.max_step-1):
            terminal = True

        truncated = False

        return next_state, reward, terminal, truncated, {}

    def get_obs(self):
        self.session.tui.file.export.ascii(self.flowfield_name, ",", "yes", "velocity-magnitude", ",", "yes")
        new_observation = pd.read_csv(self.flowfield_name, usecols=[3])
        return np.array(new_observation)

    def get_drag_lift(self):

        df = pd.read_csv(self.cdcl_path, skiprows=2, sep=' ')

        # 提取后50行数据
        last_50_rows = df.tail(50)

        # 计算"cd"列的平均值
        cd_mean = last_50_rows.iloc[:, 1].mean()

        # 计算"cl"列的绝对值的平均值
        cl_mean = last_50_rows.iloc[:, 2].abs()
        cl_mean = cl_mean.mean()
        return cd_mean, cl_mean

    def write_history_parameters(self):
        self.history_parameters["observation"].extend(self.new_observation)
        self.history_parameters["drag"].extend(np.array(self.drag))
        self.history_parameters["lift"].extend(np.array(self.lift))


    def __str__(self):
        print("Env2DCylinder ---")

    def close(self):
        """
        Close environment. No other method calls possible afterwards.
        """

        self.ready_to_use = False

    def seed(self, seed):
        """
        Sets the random seed of the environment to the given value (current time, if seed=None).
        Naturally deterministic Environments don't have to implement this method.

        Args:
            seed (int): The seed to use for initializing the pseudo-random number generator (default=epoch time in sec).
        Returns: The actual seed (int) used OR None if Environment did not override this method (no seeding supported).
        """

        # we have a deterministic environment: no need to implement

        return None

    def reset(self):
        """
        Reset environment and setup for new episode.

        Returns:
            initial state of reset environment.
        """
        if self.episode_number > 0:
            mean_accumulated_drag = self.accumulated_drag / 80
            mean_accumulated_lift = self.accumulated_lift / 80
            print("mean accumulated drag on the whole episode: {}".format(mean_accumulated_drag))

            chance = random.random()

            probability_hard_reset = 0.2

            if chance < probability_hard_reset:
                self.session.exit()
                self.start_class(complete_reset=True)
            else:
                self.start_class(complete_reset=False)

            next_state = np.transpose(np.array(self.new_observation))

            self.episode_number += 1

        else:
            self.start_class(complete_reset=True)
            self.episode_number += 1

            next_state = np.transpose(np.array(self.new_observation))

        return(next_state)


    def compute_reward(self):
        return self.drag + 0.159 - 0.2 * self.lift

    # 这里这个reward需要修改。。。the 0.159 value is a proxy value corresponding to the mean drag when no control;  may depend on the geometry






