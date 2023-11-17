import torch
from torch.optim.optimizer import Optimizer, required
from torch import Tensor
from typing import List, Optional
import copy
from torch.optim.sgd import sgd

class Lookaround(Optimizer):
 
    def __init__(self, params, lr=required, momentum=0, dampening=0,head_num = 3, frequence = 1,
                 weight_decay=0, nesterov=False, *, maximize=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, maximize=maximize, head_num = head_num, frequence = frequence)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        
        self.base_w = 0 
        self.accu_w = None
        self.net_head = []
        self.step_n = 0
        super(Lookaround, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Lookaround, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if self.step_n == 0:
                for i in range(group['head_num']):
                    self.net_head.append(copy.deepcopy(group['params']))
              
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
             
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            maximize = group['maximize']
            lr = group['lr']
            
            head = self.step_n % group['head_num']

            m_str = 'momentum_buffer_' + str(head)

            a = (self.step_n % (group['frequence'] * group['head_num'])) // group['head_num']
            r = (self.step_n % (group['frequence'] * group['head_num'])) % group['head_num']
            
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    state = self.state[p]
                    if m_str not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state[m_str])
            sgd(params_with_grad,
                  d_p_list,
                  momentum_buffer_list,
                  weight_decay=weight_decay,
                  momentum=momentum,
                  lr=lr,
                  dampening=dampening,
                  nesterov=nesterov,
                  maximize=maximize,)
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                self.state[p][m_str] = momentum_buffer
            for i,p in enumerate(group['params']):
                self.net_head[r][i].data = group['params'][i].data
            if (self.step_n % (group['frequence'] * group['head_num'])) + 1 == (group['frequence'] * group['head_num']):
                for i,p in enumerate(group['params']):
                    self.net_head[0][i][:] = 1/group['head_num'] * self.net_head[0][i][:]
                    for j in range(1,group['head_num']):
                        self.net_head[0][i][:] = self.net_head[0][i][:] + 1/group['head_num'] * self.net_head[j][i][:]
                    for j in range(1,group['head_num']):
                        self.net_head[j][i][:] = self.net_head[0][i][:] 
            for i,p in enumerate(group['params']):
                group['params'][i].data = self.net_head[(r+1)%group['head_num']][i].data

        self.step_n += 1
        return loss