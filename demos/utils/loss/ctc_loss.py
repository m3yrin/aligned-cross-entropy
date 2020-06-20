from typing import Union
import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def sequence_ctc_loss_with_logits(
    logits: torch.FloatTensor,
    logit_mask: Union[torch.FloatTensor, torch.BoolTensor],
    targets: torch.LongTensor,
    target_mask: Union[torch.FloatTensor, torch.BoolTensor],
    blank_index: torch.LongTensor
) -> torch.FloatTensor:

    # lengths : (batch_size, )
    # calculated by counting number of mask
    
    logit_lengths = (logit_mask.bool()).long().sum(1)
    target_lengths = (target_mask.bool()).long().sum(1)

    # log_logits : (T, batch_size, n_class), this kind of shape is required for ctc_loss
    log_logits = logits.log_softmax(-1).transpose(0, 1)
    targets = targets.long()

    loss = F.ctc_loss(log_logits, 
                      targets, 
                      logit_lengths, 
                      target_lengths,
                      blank=blank_index,
                      reduction='mean')
    
    if (logit_lengths < target_lengths).sum() > 0:
        print("The length of predicted alignment is shoter than target length, increase upsample factor.")
        raise Exception

    return loss