
from torch import nn
import math

def get_normalization_layer(num_features=64, norm_type=None, num_groups=None,):
    norm_type = 'layer_norm' if norm_type is None else norm_type
    num_groups = 32 if num_groups is None else num_groups
    momentum = 0.1
    eps = 1e-5
    affine = True
    track_running_stats = True
    norm_layer = None
    norm_type = norm_type.lower() if norm_type is not None else None

    if norm_type in ['batch_norm', 'batch_norm_2d']:
        norm_layer = nn.BatchNorm2d(num_features=num_features, eps=eps, momentum=momentum, affine=affine,
                                    track_running_stats=track_running_stats)
    elif norm_type == 'batch_norm_1d':
        norm_layer = nn.BatchNorm1d(num_features=num_features, eps=eps, momentum=momentum, affine=affine,
                                    track_running_stats=track_running_stats)
    elif norm_type in ['sync_batch_norm', 'sbn']:
        norm_layer = nn.SyncBatchNorm(num_features=num_features, eps=eps, momentum=momentum, affine=affine,
                                      track_running_stats=track_running_stats)
    elif norm_type in ['group_norm', 'gn']:
        num_groups = math.gcd(num_features, num_groups)
        norm_layer = nn.GroupNorm(num_channels=num_features, num_groups=1, eps=eps, affine=affine)
    elif norm_type in ['instance_norm', 'instance_norm_2d']:
        norm_layer = nn.InstanceNorm2d(num_features=num_features, eps=eps, momentum=momentum, affine=affine,
                                       track_running_stats=track_running_stats)
    elif norm_type == "instance_norm_1d":
        norm_layer = nn.InstanceNorm1d(num_features=num_features, eps=eps, momentum=momentum, affine=affine,
                                       track_running_stats=track_running_stats)
    elif norm_type in ['layer_norm', 'ln']:
        norm_layer = nn.LayerNorm(normalized_shape=num_features, eps=eps, elementwise_affine=affine)
    return norm_layer


