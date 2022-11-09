import torch
from dotenv import load_dotenv

load_dotenv()


def normalize_hand(points):
    """
    Normalize the hand points' x, y, and z to be in the range [-1, 1]

    Args:
        points list: List of 3D points (x, y, z)
    """
    # Normalize x
    x_min = torch.min(points[:, 0])
    x_max = torch.max(points[:, 0])
    x_range = x_max - x_min
    points[:, 0] = (points[:, 0] - x_min) / x_range * 2 - 1

    # Normalize y
    y_min = torch.min(points[:, 1])
    y_max = torch.max(points[:, 1])
    y_range = y_max - y_min
    points[:, 1] = (points[:, 1] - y_min) / y_range * 2 - 1

    # Normalize z
    z_min = torch.min(points[:, 2])
    z_max = torch.max(points[:, 2])
    z_range = z_max - z_min
    points[:, 2] = (points[:, 2] - z_min) / z_range * 2 - 1

    # Convert NaNs to 0
    points[torch.isnan(points)] = 0

    return points


def points_to_tensor(points, hands, device="cpu") -> torch.Tensor:
    """
    Converts the points to a tensor

    Args:
        points (List[List[Point]]): The points
        hands (List[Hand]): The hands

    Returns:
        torch.Tensor: The tensor
    """
    left = right = None

    num_hands = len(hands)
    for idx in range(num_hands):
        points_arr = points[idx]
        hand = hands[idx]

        if hand == "Left":
            left = points_arr
        elif hand == "Right":
            right = points_arr

    # Convert to tensors
    left_tensor = torch.tensor(left) if left else torch.zeros((21, 3))
    right_tensor = torch.tensor(right) if right else torch.zeros((21, 3))

    # Normalize
    left_tensor = normalize_hand(left_tensor)
    right_tensor = normalize_hand(right_tensor)

    # Concatenate
    data = torch.cat((left_tensor, right_tensor), dim=0).to(device)
    return data if (left or right) else None
