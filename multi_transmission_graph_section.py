import numpy.random
import pandas
import pandapower as pp
import pandapower.networks as pn
import numpy as np
import copy
import torch
import gym
from utils import load_variable
import pandapower.topology as pt
import dgl
from utils import ROOT_PATH
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class TransmissionSectionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, args, evaluation=False):
        if args.env_id == 'M5case118':
            self.original_net = pn.case118()
        elif args.env_id == 'M5case9241':
            self.original_net = pn.case9241pegase()
        else:
            assert False, 'env_id not exist'
        nxgraph = pt.create_nxgraph(self.original_net)
        self.graph = dgl.from_networkx(nxgraph)
        self.n_bus = self.original_net.bus.shape[0]
        self.n_gen = self.original_net.gen.shape[0]

        self.current_power_section = None
        self.evaluation = evaluation
        self.env_id = args.env_id
        self.method = args.method

        if self.env_id == 'M5case9241':
            self.n_line = 29  # for case9241, use transmission lines of interest with respective of all interfaces
        else:
            self.n_line = len(self.original_net.line)

        if not os.path.exists(os.path.join(ROOT_PATH, args.env_id, 'multi_train_control_nets.pt')):
            assert False, 'No Available Data'

        if not self.evaluation:
            self.control_nets = load_variable(os.path.join(ROOT_PATH, args.env_id, 'multi_train_control_nets.pt'))
        else:
            self.control_nets = load_variable(os.path.join(ROOT_PATH, args.env_id, 'multi_test_control_nets.pt'))

        cost = self.original_net.poly_cost
        cost['element'] = pandas.to_numeric(cost['element'], downcast='integer')
        cost.set_index(['element'], inplace=True)
        cost.sort_index(inplace=True)
        self.gen_cost = cost.loc[cost['et'] == 'gen']
        self.ext_grid_cost = cost.loc[cost['et'] == 'ext_grid']

        self.n_net = len(self.control_nets['control_nets_power_section'])
        if args.env_id == 'M5case9241':
            self.choose_num = self.control_nets['choose_num']
        self.task_num = None
        self.task_id = args.task_id

        self.total_section_lines = []
        self.total_section_trafos = []
        self.total_power_target = []
        self.total_net_idx = [[] for _ in range(10)]
        cur_section = None
        j = -1
        for i in range(self.n_net):
            section = self.control_nets['control_nets_section'][i]
            if section != cur_section:
                self.total_section_lines.append(section)
                self.total_section_trafos.append(self.control_nets['control_nets_trafos'][i])
                self.total_power_target.append(self.control_nets['control_nets_target'][i])
                j += 1
                cur_section = section
            self.total_net_idx[j].append(i)

        # for case9241, use transmission lines of interest with respective of all interfaces
        if self.env_id == 'M5case9241':
            self.section_set = np.concatenate(self.total_section_lines).tolist()

        self.task_num = None
        self.task_id = args.task_id

        if isinstance(args.task_id, int):
            self.task_num = args.task_id
        elif isinstance(args.task_id, list):
            self.task_num = len(args.task_id)
        else:
            assert False, 'task_id invalid'

        self.n_adjust_step = 2  # dispatch step
        self.adjust_ratio = np.linspace(0.9, 1.1, num=self.n_adjust_step)  # dispatch rate

        self.action_space = gym.spaces.Discrete(n=self.n_gen * self.n_adjust_step)

        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.n_bus * 4+self.task_num*self.n_line,), dtype=np.float32)

        self.current_step = 0
        self.current_idx = -1
        self.current_net_undo = None
        self.current_net = None
        self.converged = None
        self.success = None
        self.info = None

    def _get_cost(self):
        gen_power_cost = sum(self.gen_cost['cp2_eur_per_mw2'] * (self.current_net.gen['p_mw']) ** 2 +
                             self.gen_cost['cp1_eur_per_mw'] * (self.current_net.gen['p_mw']))
        total_power_cost = int(gen_power_cost +
                               sum(self.ext_grid_cost['cp2_eur_per_mw2'] * (self.current_net.res_ext_grid['p_mw']) ** 2
                               + self.ext_grid_cost['cp1_eur_per_mw'] * self.current_net.res_ext_grid['p_mw']))
        return -total_power_cost

    def _get_section_onehot(self):
        if self.env_id == 'M5case9241':
            self.section_onehot = torch.Tensor([])
            for section_line in self.section_lines:
                section_onehot = torch.zeros(len(self.section_set), dtype=torch.float32)
                for loc in section_line:
                    section_onehot[self.section_set.index(loc)] = 1
                self.section_onehot = torch.cat((self.section_onehot, section_onehot), dim=-1)
        else:
            self.section_onehot = torch.Tensor([])
            for section_line in self.section_lines:
                section_onehot = torch.zeros(self.n_line, dtype=torch.float32)
                for loc in section_line:
                    section_onehot[loc] = 1
                self.section_onehot = torch.cat((self.section_onehot, section_onehot), dim=-1)
        return self.section_onehot

    def _random_init_section(self):
        if isinstance(self.task_id, int):
            choose_num = numpy.sort(numpy.random.choice(10, size=self.task_num, replace=False))
            self.section_lines = [self.total_section_lines[i] for i in choose_num]
            self.section_trafos = [self.total_section_trafos[i] for i in choose_num]
            self.power_target = [self.total_power_target[i] for i in choose_num]
            self.ava_net_idx = numpy.concatenate([self.total_net_idx[i] for i in choose_num])
        elif isinstance(self.task_id, list):
            self.section_lines = [self.total_section_lines[i] for i in self.task_id]
            self.section_trafos = [self.total_section_trafos[i] for i in self.task_id]
            self.power_target = [self.total_power_target[i] for i in self.task_id]
            self.ava_net_idx = numpy.concatenate([self.total_net_idx[i] for i in self.task_id])

    def _random_init_section9241(self, idx):
        if isinstance(self.task_id, int):
            choose_num = self.choose_num[idx]
            self.section_lines = [self.total_section_lines[i] for i in choose_num]
            self.section_trafos = [self.total_section_trafos[i] for i in choose_num]
            self.power_target = [self.total_power_target[i] for i in choose_num]
            self.ava_net_idx = numpy.concatenate([self.total_net_idx[i] for i in choose_num])
        elif isinstance(self.task_id, list):
            self.section_lines = [self.total_section_lines[i] for i in self.task_id]
            self.section_trafos = [self.total_section_trafos[i] for i in self.task_id]
            self.power_target = [self.total_power_target[i] for i in self.task_id]
            self.ava_net_idx = numpy.concatenate([self.total_net_idx[i] for i in self.task_id])

    def reset(self):
        if self.env_id == 'M5case9241' and self.method == 'MAM':
            self.current_step = 0
            if self.evaluation:
                self.current_idx += 1
                if self.current_idx == self.n_net:
                    self.current_idx = 0
            else:
                self.current_idx = np.random.randint(0, self.n_net)
            self._random_init_section9241(self.current_idx)

        else:
            self._random_init_section()
            self.current_step = 0
            if self.evaluation:
                self.current_idx += 1
                if self.current_idx == self.n_net:
                    self.current_idx = 0
            else:
                self.current_idx = np.random.choice(self.ava_net_idx)

        self.current_net = self._load_net(self.current_idx)
        self.current_net_undo = copy.deepcopy(self.current_net)
        self.current_power_section = copy.deepcopy(self.control_nets['control_nets_power_section'][self.current_idx])
        return self._get_state()

    def set(self, idx):
        if self.env_id == 'M5case9241' and self.method == 'MAM':
            self._random_init_section9241(idx)
        else:
            self._random_init_section()
        if idx >= self.n_net:
            return None
        self.current_step = 0
        self.current_idx = idx
        self.current_net = self._load_net(self.current_idx)
        self.current_net_undo = copy.deepcopy(self.current_net)
        self.current_power_section = copy.deepcopy(self.control_nets['control_nets_power_section'][self.current_idx])
        return self._get_state()

    def _load_net(self, idx):
        net = copy.deepcopy(self.original_net)
        net.load['p_mw'] = copy.deepcopy(self.control_nets['control_nets_load_p'][idx])
        net.load['q_mvar'] = copy.deepcopy(self.control_nets['control_nets_load_q'][idx])
        net.gen['p_mw'] = copy.deepcopy(self.control_nets['control_nets_gen_p'][idx])
        return net

    def _get_state(self):
        try:
            pp.runpp(self.current_net)
            self.converged = True
        except Exception as e:
            assert isinstance(e, pp.powerflow.LoadflowNotConverged), 'Not Converged Error'
            self.converged = False
            self.current_net = self.current_net_undo
            pp.runpp(self.current_net)

        result = copy.deepcopy(self.current_net.res_bus)
        result['va_degree'] = (result['va_degree'] - np.mean(result['va_degree'])) / np.std(result['va_degree'])
        result['p_mw'] = (result['p_mw'] - np.mean(result['p_mw'])) / np.std(result['p_mw'])
        result['q_mvar'] = (result['q_mvar'] - np.mean(result['q_mvar'])) / np.std(result['q_mvar'])
        result['vm_pu'] = result['vm_pu'] - np.mean(result['vm_pu'])
        x = torch.tensor(np.array(result), dtype=torch.float32).reshape(-1)
        state = torch.cat((x, self._get_section_onehot()), dim=-1)
        return state

    def step(self, action):
        self.current_step += 1
        if action == -1:  # for debug
            pass
        else:
            self.current_net_undo = copy.deepcopy(self.current_net)
            action_gen_idx = np.floor(action / self.n_adjust_step)
            action_ratio_idx = action % self.n_adjust_step
            self.current_net.gen['p_mw'][action_gen_idx] *= self.adjust_ratio[action_ratio_idx]

        state = self._get_state()

        self.current_power_section = self._get_power_section()
        reward, done = self._get_reward_done()
        self.info = {'task_embedding': self._get_section_onehot(), 'is_converged': self.converged,
                     'is_success': self.success, 'cost': self._get_cost()}
        return state, reward, done, self.info

    def _get_reward_done(self):
        gen_cost = self._get_cost()

        self.success = True
        reward_total = []
        done = True
        if not self.converged:
            reward = -100
            done = True
            self.success = False

        else:
            for section_idx in range(len(self.section_lines)):
                if int(self.power_target[section_idx]*0.2) < self.current_power_section[section_idx] \
                        < int(self.power_target[section_idx]*1.4):
                    reward_total.append(100)
                else:
                    reward_total.append(-1 * abs(self.current_power_section[section_idx] -
                                                 self.power_target[section_idx])/500)
                    done = False
                    self.success = False
            reward = min(reward_total)
        if self.current_step > 50:
            done = True
            self.success = False

        reward += 5e-6 * gen_cost

        return reward, done

    def _get_power_section(self):
        power_section_total = []
        for idx in range(len(self.section_lines)):
            power_section = np.sum(np.abs(self.current_net.res_line.loc[self.section_lines[idx], 'p_from_mw']))
            if self.section_trafos[idx] != -1:
                power_section += np.sum(np.abs(self.current_net.res_trafo.loc[[self.section_trafos[idx]], 'p_hv_mw']))
            power_section_total.append(power_section)
        return power_section_total
