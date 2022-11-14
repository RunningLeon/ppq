from .AcademicQuantizer import (ACADEMIC_INT4_Quantizer,
                                ACADEMIC_Mix_Quantizer, ACADEMICQuantizer)
from .base import BaseQuantizer
from .DSPQuantizer import PPL_DSP_Quantizer, PPL_DSP_TI_Quantizer
from .FPGAQuantizer import FPGAQuantizer
from .MetaxQuantizer import MetaxChannelwiseQuantizer, MetaxTensorwiseQuantizer
from .MyQuantizer import ExtQuantizer
from .NCNNQuantizer import NCNNQuantizer
from .NXPQuantizer import NXP_Quantizer
from .OpenvinoQuantizer import OpenvinoQuantizer
from .ORTQuantizer import ORT_PerChannelQuantizer, ORT_PerTensorQuantizer
from .PPLQuantizer import PPLCUDAQuantizer
from .TengineQuantizer import TengineQuantizer
from .TensorRTQuantizer import TensorRTQuantizer
