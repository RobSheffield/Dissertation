import numpy as np
import torch


def _resolve_core_model(model):
	# Keep full nn.Module models intact (e.g., Ultralytics DetectionModel),
	# otherwise route through wrapper.model when present.
	if isinstance(model, torch.nn.Module):
		return model
	return model.model if hasattr(model, "model") else model


def _pick_default_hook_layers(core_model):
	modules = getattr(core_model, "model", None)
	if isinstance(modules, torch.nn.Sequential) and len(modules) >= 3:
		return [len(modules) - 3, len(modules) - 2]
	return []


def _get_layer_by_spec(core_model, layer_spec):
	modules = getattr(core_model, "model", None)
	if isinstance(layer_spec, int):
		return modules[layer_spec], f"idx:{layer_spec}"
	named = dict(core_model.named_modules())
	return named[layer_spec], layer_spec


def hook_layer(model, layer_specs=None):
	core_model = _resolve_core_model(model)
	if layer_specs is None:
		layer_specs = _pick_default_hook_layers(core_model)

	activations = {}
	handles = []
	hook_keys = []

	for spec in layer_specs:
		layer, key = _get_layer_by_spec(core_model, spec)
		hook_keys.append(key)

		def _make_hook(hook_key):
			def _hook(_, __, output):
				activations[hook_key] = output

			return _hook

		handles.append(layer.register_forward_hook(_make_hook(key)))

	return handles, activations, hook_keys


def _tensor_to_feature_matrix(t):
	if t.ndim == 1:
		return t.unsqueeze(1)
	if t.ndim == 2:
		return t
	if t.ndim >= 3:
		reduce_dims = tuple(range(2, t.ndim))
		return t.mean(dim=reduce_dims)
	return t.reshape(t.shape[0], -1)


def get_ats(model, data_loader, device, layer_specs=None):
	core_model = _resolve_core_model(model).to(device)
	core_model.eval()

	model_dtype = None
	for p in core_model.parameters():
		model_dtype = p.dtype
		break

	handles, activations, hook_keys = hook_layer(core_model, layer_specs=layer_specs)

	at_list = []
	with torch.no_grad():
		for batch in data_loader:
			if torch.is_tensor(batch):
				inputs = batch
			elif isinstance(batch, (tuple, list)):
				inputs = batch[0]
			else:
				inputs = batch["img"] if "img" in batch else batch["images"]

			if model_dtype is not None and inputs.is_floating_point():
				inputs = inputs.to(device=device, dtype=model_dtype, non_blocking=True)
			else:
				inputs = inputs.to(device, non_blocking=True)
			_ = core_model(inputs)

			batch_at = torch.cat(
				[_tensor_to_feature_matrix(activations[key]) for key in hook_keys],
				dim=1,
			)
			at_list.append(batch_at.detach().cpu().numpy())

	for h in handles:
		h.remove()

	return np.concatenate(at_list, axis=0)


__all__ = [
	"hook_layer",
	"get_ats",
]
