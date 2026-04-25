from depth_anything_v2.dpt import DepthAnythingV2
import torch
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitb'
model_path = f'checkpoints/depth_anything_v2_{encoder}.pth'

model = DepthAnythingV2(**model_configs[encoder])
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device).eval().requires_grad_(False)

def depth_loss(A: torch.Tensor, B: torch.Tensor, model=model) -> torch.Tensor:
    A = A.to(device)
    B = B.to(device)

    if A.shape[1] == 1:
        A = A.repeat(1, 3, 1, 1)
    if B.shape[1] == 1:
        B = B.repeat(1, 3, 1, 1)

    target_size = 518

    resized_A = F.interpolate(A, size=(target_size, target_size), mode='bilinear', align_corners=False)
    resized_B = F.interpolate(B, size=(target_size, target_size), mode='bilinear', align_corners=False)

    features_A = model.pretrained.get_intermediate_layers(
        resized_A,
        model.intermediate_layer_idx[model.encoder],
        return_class_token=True
    )
    features_B = model.pretrained.get_intermediate_layers(
        resized_B,
        model.intermediate_layer_idx[model.encoder],
        return_class_token=True
    )

    total_loss = 0.0
    num_layers = len(features_A)

    for feat_A, feat_B in zip(features_A, features_B):
        token_feat_A, _ = feat_A
        token_feat_B, _ = feat_B


        layer_loss = F.mse_loss(token_feat_A, token_feat_B)
        total_loss += layer_loss

    return total_loss / num_layers


if __name__ == '__main__':
    batch_size, channels, height, width = 4, 1, 1024, 1024
    dummy_input_A = torch.randn(batch_size, channels, height, width).to(device)
    dummy_input_B = torch.randn(batch_size, channels, height, width).to(device)

    loss = depth_loss(dummy_input_A, dummy_input_B, model)

    print(f"Computed depth loss: {loss.item():.6f}")

