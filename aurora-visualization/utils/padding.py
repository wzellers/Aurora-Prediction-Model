import torch.nn.functional as F

def pad_to_patch_size(data, patch_size):
    shape = data.shape
    if len(shape) == 4:
        b, c, h, w = shape
    elif len(shape) == 3:
        data = data.unsqueeze(1)
        b, c, h, w = data.shape
    else:
        raise ValueError(f"Unsupported tensor shape for padding: {shape}")

    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    if h == 0 or w == 0:
        raise ValueError(f"Attempting to pad a tensor with zero height or width: {data.shape}")
    
    padded = F.pad(data, (0, pad_w, 0, pad_h))
    return padded