# from . import generic_utils, module_utils
from .generic_utils import EMA, zero_grad
from .module_utils import view_module
from .statics_utils import get_flops_params_mems, stats_module_sparsity,stats_sparsity,get_attainable_FLOPS,ModuleIO,ModuleIOs