import torch
import math

def update_cache_with_refmap(cached_features_dict: dict, refmap: torch.Tensor) -> dict:
    if not cached_features_dict:
        return cached_features_dict

    # refmap: (1, 64, 64, 1)
    refmap_flat = refmap.view(-1).long()
    needs_ref = refmap_flat != -1
    
    if not needs_ref.any():
        return cached_features_dict
        
    ref_indices = refmap_flat[needs_ref]
    needs_ref_indices = torch.nonzero(needs_ref).squeeze(-1) # indices in 4096
    
    for key, value in cached_features_dict.items():
        if "qkv" in key and "qkvpe" not in key:
            num_windows = value.shape[1]
            num_hw = value.shape[3]
            
            if num_windows == 1 and num_hw >= 4096:
                # Global
                old_values = value[:, :, :, ref_indices, :].clone()
                value[:, :, :, needs_ref_indices, :] = old_values
                cached_features_dict[key] = value
            elif num_windows > 1:
                # Windowed
                sqrt_num_windows = int(math.sqrt(num_windows))
                sqrt_num_hw = int(math.sqrt(num_hw))
                
                key_reshaped = value.view(
                    value.shape[0], sqrt_num_windows, sqrt_num_windows, value.shape[2],
                    sqrt_num_hw, sqrt_num_hw, value.shape[4]
                )
                key_reshaped = key_reshaped.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(
                    value.shape[0], sqrt_num_windows * sqrt_num_hw, sqrt_num_windows * sqrt_num_hw, value.shape[2], value.shape[4]
                ) # (3, 70, 70, num_heads, C)
                
                # Apply refmap on the top-left 64x64 region.
                region_64x64 = key_reshaped[:, :64, :64, :, :].reshape(value.shape[0], 4096, value.shape[2], value.shape[4])
                old_values = region_64x64[:, ref_indices, :, :].clone()
                region_64x64[:, needs_ref_indices, :, :] = old_values
                key_reshaped[:, :64, :64, :, :] = region_64x64.view(value.shape[0], 64, 64, value.shape[2], value.shape[4])
                
                # Repartition
                key_reshaped = key_reshaped.view(
                    value.shape[0], sqrt_num_windows, sqrt_num_hw, sqrt_num_windows, sqrt_num_hw, value.shape[2], value.shape[4]
                )
                key_reshaped = key_reshaped.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
                key_reshaped = key_reshaped.view(*value.shape)
                cached_features_dict[key] = key_reshaped
                
        elif "attn_v" in key:
            old_values = value[:, ref_indices, :].clone()
            value[:, needs_ref_indices, :] = old_values
            cached_features_dict[key] = value
            
        elif "attn" in key:
            old_values = value[:, ref_indices, :].clone()
            value[:, needs_ref_indices, :] = old_values
            
            old_values_key = value[:, :, ref_indices].clone()
            value[:, :, needs_ref_indices] = old_values_key
            cached_features_dict[key] = value
            
        elif "out" in key:
            B, H, W, C = value.shape
            value_flat = value.view(B, H * W, C)
            old_values = value_flat[:, ref_indices, :].clone()
            value_flat[:, needs_ref_indices, :] = old_values
            cached_features_dict[key] = value_flat.view(B, H, W, C)

    return cached_features_dict

cached_features = {
    "block0_qkv": torch.randn(3, 25, 16, 196, 64),
    "block2_qkv": torch.randn(3, 1, 16, 4096, 64)
}

refmap = torch.randint(-1, 4096, (1, 64, 64, 1))
res = update_cache_with_refmap(cached_features, refmap)
print("Shapes:")
print(res["block0_qkv"].shape, res["block2_qkv"].shape)
