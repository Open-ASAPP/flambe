"""
Intergation of the pytorch_transformers optimization module
"""

import pytorch_transformers as pt

from flambe import Component


class ConstantLRSchedule(Component, pt.ConstantLRSchedule):
    pass


class WarmupConstantSchedule(Component, pt.WarmupConstantSchedule):
    pass


class WarmupLinearSchedule(Component, pt.WarmupLinearSchedule):
    pass


class AdamW(Component, pt.AdamW):
    pass
