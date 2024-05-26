import math
import random
import torch
import torch.nn as nn  
import torchvision.datasets as datasets
    

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


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
    
    def map_ind(self, ind_dset, ind_ba_1, ind_ba_2):
        """ Map the in-batch indices to the original indices of dataset. """
        
        def get_dset_key(ind_ba):
            ind_ba = ind_ba.detach().clone()
            set0 = ind_dset[ind_ba]
            if torch.distributed.get_world_size() > 1:
                set0 = concat_all_gather(set0)
            return set0.cpu().numpy().tolist()

        set1 = get_dset_key(ind_ba_1)
        set2 = get_dset_key(ind_ba_2)
        set12 = set(set1) & set(set2)
        
        if len(set12) < 0.9 * len(set1):
            rest_len = len(set1) - len(set12)
            rest_set = (set(set1) | set(set2)) - set12
            set12.update(set(random.sample(rest_set, k=rest_len)))
            
        return set12
        
       
    def forward(self, ind_dset=None, losses: dict = None):
        """ Gather the metric from all workers (the inputs are not reduced!). """       
        loss1, loss2 = losses['loss1'], losses['loss2']        
        
        # get the in-batch indices of the instances that satisfy the conditions
        ind_ill_1, ind_red_1 = self.get_ind(loss1)
        ind_ill_2, ind_red_2 = self.get_ind(loss2)
            
        # get the intersection of the two sets that both satisfy the conditions
        ind_dset = ind_dset.detach().clone()
        
        red_set = self.map_ind(ind_dset, ind_red_1, ind_red_2)
        ill_set = self.map_ind(ind_dset, ind_ill_1, ind_ill_2)
        
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
    

class ImageFolderCustom(datasets.ImageFolder):
    def __init__(self, root, 
                 transform=None, 
                 target_transform=None, 
                 loader=datasets.folder.default_loader, 
                 prune_set=None):
        super(ImageFolderCustom, self).__init__(root, 
                                                transform=transform, 
                                                target_transform=target_transform, 
                                                loader=loader)
        self.prune_set = prune_set
        self.length = len(self.samples)
        
    def reset_reduction(self):
        self.prune_set = None
        self.length = len(self.samples)
        
    def set_length(self, length):
        self.length = length
        
    def __len__(self):
        return self.length
        
    def set_prune_set(self, prune_set):
        self.prune_set = prune_set
        set_org = set(range(len(self.samples)))
        set_aft = set_org - set(prune_set)
        
        self.ind_map = dict(zip(range(len(set_aft)), list(set_aft)))
        
        return len(set_aft)
    
    def __getitem__(self, idx):
        if hasattr(self, 'ind_map') and not self.prune_set is None:
            idx = self.ind_map[idx]
        
        path, target = self.samples[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, idx
    