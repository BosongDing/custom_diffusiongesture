from typing import Dict, Iterable, Tuple
import torch


def set_requires_grad(module: torch.nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def build_param_groups(model, lr_audio: float, lr_diff: float, lr_dec: float, weight_decay: float):
    groups = [
        {"params": list(model.audio_encoder.parameters()), "lr": lr_audio, "weight_decay": weight_decay, "name": "audio"},
        {"params": list(model.diffusion.net.parameters()), "lr": lr_diff, "weight_decay": weight_decay, "name": "diff"},
    ]
    for name, dec in model.decoders.items():
        groups.append({"params": list(dec.parameters()), "lr": lr_dec, "weight_decay": weight_decay, "name": f"dec_{name}"})
    return groups


def apply_mode_freeze(model, mode: str):
    if mode == 'frozen':
        for mdl in model.models.values():
            # freeze full AE/VAE
            set_requires_grad(mdl, False)
        for dec in model.decoders.values():
            set_requires_grad(dec, False)
        set_requires_grad(model.audio_encoder, True)
        set_requires_grad(model.diffusion, True)
        return

    # default start: enc/dec frozen, audio+diff trainable
    for mdl in model.models.values():
        set_requires_grad(mdl, False)
    for dec in model.decoders.values():
        set_requires_grad(dec, False)
    set_requires_grad(model.audio_encoder, True)
    set_requires_grad(model.diffusion, True)


def step_based_unfreeze(step: int, s1: int, s2: int, s3: int, model):
    # s2: unfreeze encoders (full model sans decoder)
    if step >= s2:
        for name, mdl in model.models.items():
            # unfreeze the encoder part if accessible
            if hasattr(mdl, 'encoder'):
                set_requires_grad(mdl.encoder, True)
    # s3: unfreeze decoders
    if step >= s3:
        for dec in model.decoders.values():
            set_requires_grad(dec, True)


def isolated_window_control(global_step: int, windows: Tuple[int, int, int], model, optimizer: torch.optim.Optimizer):
    w1, w2, w3 = windows
    if global_step < w1:
        set_requires_grad(model.audio_encoder, True)
        set_requires_grad(model.diffusion, False)
        for mdl in model.models.values():
            set_requires_grad(mdl, False)
        for dec in model.decoders.values():
            set_requires_grad(dec, False)
    elif global_step < w1 + w2:
        set_requires_grad(model.audio_encoder, False)
        set_requires_grad(model.diffusion, True)
        for mdl in model.models.values():
            set_requires_grad(mdl, False)
        for dec in model.decoders.values():
            set_requires_grad(dec, False)
    elif global_step < w1 + w2 + w3:
        set_requires_grad(model.audio_encoder, False)
        set_requires_grad(model.diffusion, False)
        for mdl in model.models.values():
            set_requires_grad(mdl, False)
        for dec in model.decoders.values():
            set_requires_grad(dec, True)
    else:
        set_requires_grad(model.audio_encoder, False)
        set_requires_grad(model.diffusion, False)
        for mdl in model.models.values():
            set_requires_grad(mdl, False)
        for dec in model.decoders.values():
            set_requires_grad(dec, True) 