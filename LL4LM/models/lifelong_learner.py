import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


class LifelongLearner(nn.Module):
    
    def __init__(self,
                 base_model: nn.Module,
                 device: torch.device):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(self.base_model.config.hidden_dropout_prob)
        self.head = nn.Linear(self.base_model.config.hidden_size, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.device = device
        self.to(device)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.base_model(input_ids, attention_mask, token_type_ids, 
                                  return_dict=True)
        pooler_output = outputs['pooler_output']
        logits = self.head(pooler_output)
        outputs['logits'] = logits
        return outputs

    def step(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        labels = batch.pop("label")
        outputs = self.forward(**batch)
        # logits = torch.squeeze(outputs['logits'])
        # loss = self.loss_fn(logits.float(), labels.float())
        logits = outputs['logits']
        loss = self.loss_fn(logits.float(), labels.unsqueeze(1).float())
        labels = labels.detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        predictions = logits >= 0.0 # probability above 0.5
        accuracy = accuracy_score(labels, predictions)
        return loss, accuracy

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)