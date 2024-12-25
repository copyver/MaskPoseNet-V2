import torch


def get_default_input_shape(cfg):
    """
    Get the default input shape of the model.
    """
    b = cfg.TEST_DATA.BATCH_SIZE
    return {
        'pts': (b, cfg.TEST_DATA.N_SAMPLE_OBSERVED_POINT, 3),
        'rgb': (b, 3, cfg.TEST_DATA.IMG_SIZE, cfg.TEST_DATA.IMG_SIZE),
        'rgb_choose': (b, cfg.TEST_DATA.N_SAMPLE_OBSERVED_POINT),
        'model': (b, cfg.TEST_DATA.N_SAMPLE_MODEL_POINT, 3),
        'dense_po': (b, cfg.POSE_MODEL.FINE_NPOINT, 3),
        'dense_fo': (b, cfg.POSE_MODEL.FINE_NPOINT, cfg.POSE_MODEL.FEATURE_EXTRACTION.OUT_DIM),
    }


def get_default_tensor(input_dict, device='cuda'):
    """
    Initialize tensors based on the default input shape dictionary and move them to the specified device.

    Args:
        fp16:
        input_dict: (dict): The default input shape of the model
        device : The device to which the tensors should be moved (e.g., 'cuda' or 'cpu').

    Returns:
        dict: A dictionary of initialized tensors on the specified device.
    """
    # Get the default input shapes from the provided function

    # Initialize tensors with zeros and move them to the specified device
    torch.float
    for key, shape in input_dict.items():
        # 简单使用 torch.empty 创建占位张量
        # 逻辑上各输入代表的含义需根据实际模型做对应dummy值生成
        # 此处仅作为warmup占位，不影响实际推理结果
        if 'choose' in key:
            # choose 通常是索引或int类型张量，使用long类型填充
            input_dict[key] = torch.zeros(shape, dtype=torch.long, device=device)
        else:
            input_dict[key] = torch.empty(shape, dtype=torch.float, device=device)

    return input_dict


