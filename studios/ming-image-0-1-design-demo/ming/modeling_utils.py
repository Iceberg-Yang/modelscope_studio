"""Minimal subset of the reference repo's modeling_utils.py.

The upstream file additionally imports `from whisper.model import AudioEncoder`,
which is only needed for the audio path.  We keep just the two helpers that
`modeling_bailing_moe_v2.py` imports, so the audio stack does not have to be
vendored.
"""

import torch


def patch_continuous_features(
    input_embeddings: torch.Tensor,
    placeholder_loc_lens: torch.Tensor,
    encoded_feats: torch.Tensor,
    encoded_feat_lens: torch.Tensor,
):
    batch_size = input_embeddings.size(0)
    for i in range(batch_size):
        audio_feat_start = 0
        for j in range(placeholder_loc_lens.shape[1]):
            placeholder_start: int = int(placeholder_loc_lens[i, j, 0].item())
            placeholder_len: int = int(placeholder_loc_lens[i, j, 1].item())
            if placeholder_len <= 0:
                break
            feat_len = int(encoded_feat_lens[i, j].item())
            real_feat_len = feat_len
            if feat_len > placeholder_len:
                feat_len = placeholder_len
            target_len = min(feat_len, placeholder_len)
            input_embeddings[i, placeholder_start:placeholder_start + target_len] = encoded_feats[
                i, audio_feat_start:audio_feat_start + target_len
            ]
            audio_feat_start += real_feat_len
    return input_embeddings


def build_modality_mask(placeholder_loc_lens: torch.Tensor, shape: torch.Size):
    mask = torch.zeros(shape, dtype=torch.bool)
    for i in range(placeholder_loc_lens.shape[0]):
        for j in range(placeholder_loc_lens.shape[1]):
            start: int = int(placeholder_loc_lens[i, j, 0].item())
            length: int = int(placeholder_loc_lens[i, j, 1].item())
            if length <= 0:
                break
            mask[i, start:start + length] = True
    return mask
