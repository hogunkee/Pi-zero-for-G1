import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_g1_example() -> dict:
    """Creates a random input example for the G1 policy."""
    return {
        "observation/rs_view": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_arm": np.random.rand(7),
        "observation/right_arm": np.random.rand(7),
        "observation/left_hand": np.random.rand(7),
        "observation/right_hand": np.random.rand(7),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class G1Inputs(transforms.DataTransformFn):

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # data: ['actions', 'observation/image', 'prompt', 'state']
        # data['actions']: (16,31)
        # data['state']: (43)
        # data['prompt']: 'Pick up the cup and place it on the plate'
        # data['observation/image']: (3, 480, 640)
        # First, concatenate the joints and gripper into the state vector.
        # state = np.concatenate([data["left_arm"], data["right_arm"], data["left_hand"], data["right_hand"]])

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        base_image = _parse_image(data["observation/image"])

        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        images = (base_image, np.zeros_like(base_image), np.zeros_like(base_image))
        image_masks = (np.True_, np.False_, np.False_)
        inputs = {
            "state": data['state'],
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class G1Outputs(transforms.DataTransformFn):

    def __call__(self, data: dict) -> dict:
        # 31 dof actions for 2 arms and 2 hands and 1 waist
        return {"actions": np.asarray(data["actions"][:, :31])}
		# 28 dof actions for 2 arms and 2 hands
        # return {"actions": np.asarray(data["actions"][:, :28])}
