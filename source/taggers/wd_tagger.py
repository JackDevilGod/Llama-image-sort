import timm
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from PIL import Image
from torch import nn, Tensor
from torch.nn import functional as F
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform


@dataclass
class LabelData:
    names: list[str]
    rating: list[int]
    general: list[int]
    character: list[int]


def load_labels_hf(
    repo_id: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> LabelData:
    try:
        csv_path = hf_hub_download(
            repo_id=repo_id,
            filename="selected_tags.csv",
            revision=revision,
           token=token
        )
        csv_path = Path(csv_path).resolve()
    except HfHubHTTPError as e:
        raise FileNotFoundError(f"selected_tags.csv failed to download from {repo_id}") from e

    usecols = ["name", "category"]
    df: pd.DataFrame = pd.read_csv(csv_path, usecols=usecols)
    rating = list(np.where(df["category"] == 9)[0])
    general = list(np.where(df["category"] == 0)[0])
    character = list(np.where(df["category"] == 4)[0])
    tag_data = LabelData(
        names=df["name"].tolist(),
        rating=rating,
        general=general,
        character=character,
    )

    return tag_data


def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    # convert to RGB/RGBA if not already (deals with palette images etc.)
    if image.mode not in ["RGB", "RGBA"]:
        if "transparency" in image.info:
            image = image.convert("RGBA")
        else:
            image = image.convert("RGB")
    # convert RGBA to RGB with white background
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    # get the largest dimension so we can pad to a square
    px = max(image.size)
    # pad to square with white background
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas


def get_tags(
    probs: Tensor,
    labels: LabelData,
    gen_threshold: float,
    char_threshold: float,
) -> tuple[str, str, dict[str, float], dict[str, float], dict[str, float]]:
    # Convert indices+probs to labels
    probs_list = list(zip(labels.names, probs.numpy()))

    # First 4 labels are actually ratings
    rating_labels = dict([probs_list[i] for i in labels.rating])

    # General labels, pick any where prediction confidence > threshold
    gen_labels = [probs_list[i] for i in labels.general]
    gen_labels = dict([x for x in gen_labels if x[1] > gen_threshold])
    gen_labels = dict(sorted(gen_labels.items(), key=lambda item: item[1], reverse=True))

    # Character labels, pick any where prediction confidence > threshold
    char_labels = [probs_list[i] for i in labels.character]
    char_labels = dict([x for x in char_labels if x[1] > char_threshold])
    char_labels = dict(sorted(char_labels.items(), key=lambda item: item[1], reverse=True))

    # Combine general and character labels, sort by confidence
    combined_names = [x for x in gen_labels]
    combined_names.extend([x for x in char_labels])

    # Convert to a string suitable for use as a training caption
    caption = ", ".join(combined_names)
    taglist = caption.replace("_", " ")
    taglist = taglist.replace("(", r"\(")
    taglist = taglist.replace(")", r"\)")

    return caption, taglist, rating_labels, char_labels, gen_labels


class WDTagger:
    def __init__(self, model_repo: str) -> None:
        torch_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.torch_device: torch.device = torch_device

        model_repo_str = "hf-hub:" + model_repo
        self.model: nn.Module = (
            timm.create_model(model_repo_str).eval()
        )
        state_dict = timm.models.load_state_dict_from_hf(model_repo)
        self.model.load_state_dict(state_dict)

        self.labels: LabelData = load_labels_hf(repo_id=model_repo)

        self.transform = create_transform(
            **resolve_data_config(self.model.pretrained_cfg, model=self.model)
        )

    def get_tags(self, image_path: Path) -> list[str]:
        img_input: Image.Image = Image.open(image_path)
        img_input = pil_ensure_rgb(img_input)
        img_input = pil_pad_square(img_input)
        inputs: Tensor = self.transform(img_input).unsqueeze(0)
        inputs = inputs[:, [2, 1, 0]]

        with torch.inference_mode():
            if self.torch_device.type != "cpu":
                self.model = self.model.to(self.torch_device)
                inputs = inputs.to(self.torch_device)

            outputs = self.model.forward(inputs)

            outputs = F.sigmoid(outputs)
            # move inputs, outputs, and model back to cpu if we were on GPU
            if self.torch_device.type != "cpu":
                inputs = inputs.to("cpu")
                outputs = outputs.to("cpu")
                self.model = self.model.to("cpu")

        caption, taglist, ratings, character, general = get_tags(
            probs=outputs.squeeze(0),
            labels=self.labels,
            gen_threshold=0.35,
            char_threshold=0.75,
        )
        
        final_tags = []
        final_tags += [_.strip() for _ in taglist.split(",")]
        return final_tags


if __name__ == "__main__":
    wdtagger = WDTagger("SmilingWolf/wd-vit-tagger-v3")
    
    tags = wdtagger.get_tags(Path(r"D:\Pictures\vlcsnap-2026-02-07-22h47m08s252.png"))
    print(tags)
