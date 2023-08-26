# from . import diff_tricks,quantization,fixedq
from .diff_tricks import STE, PWL, PWL_C, get_diff_func
from .quantization import Quantization
from .fixedq import FixedQuant,FixedQuantAct
from .ternary import Ternary, TernaryAct
from .ul2q import uL2Q, uL2QAct
from .vecq import VecQ, VecQAct
from .zoomq import ZoomQ, ZoomQAct
from .clipq import ClipQ, ClipQAct
from .potq import PotQ, PotQAct
from .resq import ResQ, ResQAct