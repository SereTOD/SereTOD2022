# Copyright 2022 SereTOD Challenge Organizers
# Authors: Hao Peng (peng-h21@mails.tsinghua.edu.cn)
# Apache 2.0
import pdb
import torch 
import torch.nn as nn 

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


class ClassificationHead(nn.Module):
    def __init__(self, config):
        super(ClassificationHead, self).__init__()
        scale = 2 if config.aggregation=="dm" else 1
        self.classifier = nn.Linear(config.hidden_size*scale, config.num_labels)

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