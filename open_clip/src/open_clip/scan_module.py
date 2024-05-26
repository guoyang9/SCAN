import math
import torch
import random
import torch.nn as nn  
    
    
class PrePrune(nn.Module):
    """ We omit the use of horovod in this implementation. 
        Normally, (rho_max+rho_min) of the instances will be selected as candidates for next epoch.
        We can also use the pre-defined values (absolute variables) to filter the instances.
    """
    def __init__(self, 
                 abs_max=1e8,
                 abs_min=0.0,
                 rho_max=0.25,
                 rho_min=0.25,
                 ):   
        super(PrePrune, self).__init__()    
        self.abs_max = abs_max
        self.abs_min = abs_min
        
        # these two ratios might be automatically changed according to prune mode
        self.rho_max = rho_max
        self.rho_min = rho_min
        
    def get_ind(self, loss):
        """ In-batch indices. """
        if self.abs_max < 1e8 or self.abs_min > 0.0:
            ind_ill = torch.nonzero(loss > self.abs_max).view(-1)
            ind_red = torch.nonzero(loss < self.abs_min).view(-1)
        else:
            sorted_ls = torch.argsort(loss, descending=True)
            ind_ill = sorted_ls[:int(len(sorted_ls)*self.rho_max)]
            ind_red = sorted_ls[-int(len(sorted_ls)*self.rho_min):]
            
        return ind_ill, ind_red
    
    def map_ind(self, ind_sd, tokens, ind_ba_i, ind_ba_t):
        """ Map the in-batch indices to the original (shard id, text tokens). """
        
        def get_sd_key(ind_ba):
            ind_ba = ind_ba.detach().clone().cpu()
            return [(id_key, tuple(token)) for id_key, token in zip(
                ind_sd[ind_ba].tolist(), tokens[ind_ba].tolist())]
            
        i_set = get_sd_key(ind_ba_i)
        t_set = get_sd_key(ind_ba_t)
        it_set = set(i_set) & set(t_set)            

        # if the intersection is too small, we randomly add more instances
        if len(it_set) < 0.9 * len(i_set):
            rest_len = len(i_set) - len(it_set)
            rest_set = (set(i_set) | set(t_set)) - it_set
            it_set.update(set(random.sample(rest_set, k=rest_len)))
       
        return it_set
    
    def forward(self, ind_sd=None, tokens=None, losses: dict = None):
        """ Gather the inputs from all workers (the three inputs are already reduced). 
            We use the tuple of (shard id, text tokens) as the key for searching (compromised solution).
        """       
        loss_i, loss_t = losses['i2t'], losses['t2i']        

        # get the in-batch indices of the instances that satisfy the conditions
        ind_ill_i, ind_red_i = self.get_ind(loss_i)
        ind_ill_t, ind_red_t = self.get_ind(loss_t)       
            
        ind_sd = ind_sd.detach().clone().cpu()
        tokens = tokens[:, :5].detach().clone().cpu()
        
        # get the intersection of the image and text sets for red and ill individually
        red_set = self.map_ind(ind_sd, tokens, ind_red_i, ind_red_t)
        ill_set = self.map_ind(ind_sd, tokens, ind_ill_i, ind_ill_t)

        return red_set, ill_set 
        
        
class CosineAnnealing(nn.Module):
    def __init__(self, 
                 T_cos=3, 
                 eta_min=0,
                 rho_init=1.0):
        super(CosineAnnealing, self).__init__()
        self.T_cos = T_cos
        self.eta_min = eta_min
        self.rho_init = rho_init
        
    def forward(self, epoch):
        epoch %= (self.T_cos + 1)
        decay_ratio = self.eta_min + self.rho_init * (0.5 * (
            1 + math.cos(torch.tensor(
            (self.T_cos - epoch) / self.T_cos * math.pi))))
        return decay_ratio
    