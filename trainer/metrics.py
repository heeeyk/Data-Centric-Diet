import torch
import torch.nn.functional as F


class DiceCoefficient:
    """Computes Dice Coefficient.
    Generalized to multiple channels by computing per-channel Dice Score
    Input is expected to be probabilities instead of logits.
    This metric is mostly useful when channels contain the same semantic class (e.g. affinities computed with different offsets).
    DO NOT USE this metric when training with DiceLoss, otherwise the results will be biased towards the loss.
    """

    def __init__(self, epsilon=1e-5, ignore_index=None, num_classes=2, **kwargs):
        self.epsilon = epsilon
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def __call__(self, input, target):
        """
        :param input: 5D probability maps torch tensor (NxCxDxHxW)
        :param target: 4D or 5D ground truth torch tensor. 4D (NxDxHxW) tensor will be expanded to 5D as one-hot
        :return: Soft Dice Coefficient averaged over all channels/classes
        """
        # Average across channels in order to get the final score
        return torch.mean(compute_per_channel_dice(input, target, num_classes=self.num_classes, epsilon=self.epsilon, ignore_index=self.ignore_index))


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 5D output image (NxCxDxHxW)
    """
    if input.dim() == 3:
        # expand the input tensor to Nx1xDxHxW before scattering
        input = input.unsqueeze(0).unsqueeze(0)
    elif input.dim() == 4:
        input = input.unsqueeze(1)
    # create result tensor shape (NxCxDxHxW)
    shape = list(input.size())
    shape[1] = C
    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)


def compute_per_channel_dice(input, target, num_classes=2, epsilon=1e-5, ignore_index=None, weight=None):
    input = input.long()
    target = target.long()
    # assumes that input is a normalized probability
    assert input.dim() in [3, 4] and target.dim() in [3, 4]
    input = expand_as_one_hot(input, C=num_classes, ignore_index=ignore_index)
    target = expand_as_one_hot(target, C=num_classes, ignore_index=ignore_index)

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape. Input shape: {}, target shape: {}".format(input.shape, target.shape)

    # mask ignore_index if present
    if ignore_index is not None:
        mask = target.clone().ne_(ignore_index)
        mask.requires_grad = False

        input = input * mask
        target = target * mask

    input = flatten(input)[1:]
    target = flatten(target)[1:]

    target = target.float()
    # Compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    denominator = (input + target).sum(-1)
    return 2. * intersect / denominator.clamp(min=epsilon)


def cross_entropy_3d(x, y, w=None):
  assert len(x.shape) == len(y.shape)
  n, c, _, _, _ = x.size()
  x_t = torch.transpose(torch.transpose(torch.transpose(x, 1, 2), 2, 3), 3, 4).contiguous().view(-1, c)
  y_t = torch.transpose(torch.transpose(torch.transpose(y, 1, 2), 2, 3), 3, 4).contiguous().view(-1).long()
  loss = F.cross_entropy(x_t, y_t, weight=w)
  return loss


if __name__ == '__main__':
    # a = torch.LongTensor(16, 64, 64, 64).random_(0, 9)
    # b = expand_as_one_hot(a, C=9)
    # print(b.shape)


    a = torch.ones((1, 64, 64, 64))
    b = torch.ones((20, 64, 64, 64))*0

    pred = torch.cat((a, b), dim=0)

    a = torch.ones((2, 64, 64, 64))
    b = torch.ones((19, 64, 64, 64))*0
    label_batch = torch.cat((a, b), dim=0)
    dice_criterion = DiceCoefficient(num_classes=2)
    d = dice_criterion(pred.cpu().data, label_batch.cpu().data)


