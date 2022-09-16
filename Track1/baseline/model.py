# Copyright 2022 SereTOD Challenge Organizers
# Authors: Hao Peng (peng-h21@mails.tsinghua.edu.cn)
# Apache 2.0
import pdb
import torch 
import torch.nn as nn 
import torch.nn.functional as F

from typing import Tuple, Dict, Optional
from crf import CRF


def get_model(model_args, backbone):
    if model_args.paradigm == "token_classification":
        return ModelForTokenClassification(model_args, backbone)
    elif model_args.paradigm == "sequence_labeling":
        return ModelForSequenceLabeling(model_args, backbone)
    else:
        raise ValueError("No such paradigm")


def select_cls(hidden_states: torch.Tensor) -> torch.Tensor:
    """Select CLS token as for textual information.
    
    Args:
        hidden_states: Hidden states encoded by backbone. shape: [batch_size, max_seq_length, hidden_size]

    Returns:
        hidden_state: Aggregated information. shape: [batch_size, hidden_size]
    """
    return hidden_states[:, 0, :]


def max_pooling(hidden_states: torch.Tensor) -> torch.Tensor:
    """Applies the max-pooling operation over the sentence representation.

    Applies the max-pooling operation over the representation of the entire input sequence to capture the most useful
    information. The operation processes on the hidden states, which are output by the backbone model.

    Args:
        hidden_states (`torch.Tensor`):
            A tensor representing the hidden states output by the backbone model.

    Returns:
        pooled_states (`torch.Tensor`):
            A tensor represents the max-pooled hidden states, containing the most useful information of the sequence.
    """
    batch_size, seq_length, hidden_size = hidden_states.size()
    pooled_states = F.max_pool1d(input=hidden_states.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
    return pooled_states


class ClassificationHead(nn.Module):
    def __init__(self, config):
        super(ClassificationHead, self).__init__()
        self.classifier = nn.Linear(config.hidden_size*config.head_scale, config.num_labels)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Classify hidden_state to label distribution.
        
        Args:
            hidden_state: Aggregated textual information. shape: [batch_size, ..., hidden_size]
        
        Returns:
            logits: Raw, unnormalized scores for each label. shape: [batch_size, ..., num_labels]
        """
        logits = self.classifier(hidden_state)
        return logits


class ModelForTokenClassification(nn.Module):
    """Bert model for token classification."""

    def __init__(self, config, backbone):
        super(ModelForTokenClassification, self).__init__()
        self.backbone = backbone 
        # self.aggregation = DynamicPooling(config)
        if config.aggregation == "cls":
            self.aggregation = select_cls
        elif config.aggregation == "max_pooling":
            self.aggregation = max_pooling
        else:
            self.aggregation = select_cls
        self.cls_head = ClassificationHead(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
        ) -> Dict[str, torch.Tensor]:
        # backbone encode 
        outputs = self.backbone(input_ids=input_ids, \
                                attention_mask=attention_mask, \
                                token_type_ids=token_type_ids,
                                return_dict=True)   
        hidden_states = outputs.last_hidden_state
        # aggregation 
        # hidden_state = self.aggregation.select_cls(hidden_states)
        hidden_state = self.aggregation(hidden_states)
        # classification
        logits = self.cls_head(hidden_state)
        # compute loss 
        loss = None 
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return dict(loss=loss, logits=logits)


class ModelForSequenceLabeling(nn.Module):
    """Bert model for token classification."""

    def __init__(self, config, backbone):
        super(ModelForSequenceLabeling, self).__init__()
        self.backbone = backbone 
        self.crf = CRF(config.num_labels, batch_first=True)
        self.cls_head = ClassificationHead(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # backbone encode 
        outputs = self.backbone(input_ids=input_ids, \
                                attention_mask=attention_mask, \
                                token_type_ids=token_type_ids,
                                return_dict=True)   
        hidden_states = outputs.last_hidden_state
        # classification
        logits = self.cls_head(hidden_states) # [batch_size, seq_length, num_labels]
        # compute loss 
        loss = None 
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            # CRF
            # mask = labels != -100
            # mask[:, 0] = 1
            # labels[:, 0] = 0
            # labels = labels * mask.to(torch.long)
            # loss = -self.crf(emissions=logits, 
            #                 tags=labels,
            #                 mask=mask,
            #                 reduction = "token_mean")
        else:
            # preds = self.crf.decode(emissions=logits, mask=mask)
            # logits = torch.LongTensor(preds)
            pass 

        return dict(loss=loss, logits=logits)