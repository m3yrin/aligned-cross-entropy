from typing import Union, List
import logging
import torch
import torch.nn.functional as F

from axe import axe_loss

# for debug
torch.set_printoptions(linewidth=5000)

logger = logging.getLogger(__name__)

def sequence_axe_loss_with_logits(
    logits: torch.FloatTensor,
    logit_mask: Union[torch.FloatTensor, torch.BoolTensor],
    targets: torch.LongTensor,
    target_mask: Union[torch.FloatTensor, torch.BoolTensor],
    blank_index: torch.LongTensor,
    label_smoothing: float = None,
    delta: int = 2.0
) -> torch.FloatTensor:

    # lengths : (batch_size, )
    # calculated by counting number of masks
    logit_lengths = logit_mask.long().sum(1)
    target_lengths = target_mask.long().sum(1)
    
    loss, a = axe_loss(logits,
                       logit_lengths, 
                       targets, 
                       target_lengths,
                       blank_index = blank_index,
                       delta = 2.0,
                       reduction = 'mean',
                       label_smoothing = label_smoothing,
                       return_a = True)
    
    return loss
