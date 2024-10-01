from .CL import CL, CL_FT, CL_KL, CL_FT_epoch
from .FT import FT, FT_l1
from .GA import GA, GA_FT, GA_KL, GA_FT_epoch, GA_KL_epoch
from .KL import KL
from .RL import RL

from .NPO import NPO
def get_unlearn_method(name, *args, **kwargs):
    if name == "FT":
        unlearner = FT(*args, **kwargs)
    elif name == "l1sparse":
        unlearner = FT_l1(*args, **kwargs)
    elif name == "GA":
        unlearner = GA(*args, **kwargs)
    elif name == "GA+FT":
        unlearner = GA_FT(*args, **kwargs)
    elif name == "GA+KL":
        unlearner = GA_KL(*args, **kwargs)
    elif name == "GA_FT_epoch":
        unlearner = GA_FT_epoch(*args, **kwargs)
    elif name == "GA_KL_epoch":
        unlearner = GA_KL_epoch(if_kl=True, *args, **kwargs)
    elif name == "RL":
        unlearner = RL(*args, **kwargs)
    elif name == "KL":
        unlearner = KL(if_kl=True, *args, **kwargs)
    elif name == "CL":
        unlearner = CL(*args, **kwargs)
    elif name == "CL+FT":
        unlearner = CL_FT(if_kl=True, *args, **kwargs)
    elif name == "CL+KL":
        unlearner = CL_KL(if_kl=True, *args, **kwargs)
    elif name == "CL_FT_epoch":
        unlearner = CL_FT_epoch(*args, **kwargs)
    elif name == "NPO":
        unlearner = NPO(if_kl=True,*args, **kwargs)
    else:
        raise ValueError("No unlearning method")

    return unlearner
