import torch
import torch.nn as nn
import torch.nn.functional as F


class DebugLayer(nn.Module):
  def forward(self, x):
    print(x.shape)
    return x


class Flatten(nn.Module):
  def forward(self, x):
    batch_size = x.shape[0]
    return x.view(batch_size, -1)

def init_weights(m):
  if type(m) == nn.Linear:
      nn.init.xavier_normal_(m.weight)
      nn.init.constant_(m.bias, 0)

  if type(m) == nn.BatchNorm1d:
      nn.init.constant_(m.weight, 1)
      nn.init.constant_(m.bias, 0)
