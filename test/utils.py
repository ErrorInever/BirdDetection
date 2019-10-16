import torch


def test_collate_fn(batch):
    return tuple(*batch)


def frame_to_tensor(frame, device):
    """ convert frame to tensor"""
    frame_tensor = torch.from_numpy(frame).float() / 255.0
    # [channels, height, width]
    frame_tensor = frame_tensor.permute(2, 0, 1)
    frame_tensor = frame_tensor.to(device)
    return [frame_tensor]