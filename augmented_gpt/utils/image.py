from typing import Literal
from augmented_gpt.augmented_gpt import AugmentedGPT
from augmented_gpt.message import ContentPartImage, ContentPartText, Message, Role

import openai
from pathlib import Path
import os
from openai._types import NOT_GIVEN
from typing import overload
import base64


@overload
async def generate(
    prompt: str,
    model: Literal["dall-e-2"] = "dall-e-2",
    response_format: Literal["url", "b64_json"] = "url",
    size: Literal[
        "256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"
    ] = "1024x1024",
    style: Literal["vivid", "natural"] = "vivid",
    hd: Literal[False] = False,
    api_key: str | None = None,
) -> str: ...


@overload
async def generate(
    prompt: str,
    model: Literal["dall-e-3"],
    response_format: Literal["url", "b64_json"] = "url",
    size: Literal["1024x1024", "1792x1024", "1024x1792"] = "1024x1024",
    style: Literal["vivid", "natural"] = "vivid",
    hd: Literal[False] = False,
    api_key: str | None = None,
) -> str: ...


async def generate(
    prompt: str,
    model: Literal["dall-e-2", "dall-e-3"] = "dall-e-2",
    response_format: Literal["url", "b64_json"] = "url",
    size: Literal[
        "256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"
    ] = "1024x1024",
    style: Literal["vivid", "natural"] = "vivid",
    hd: bool = False,
    api_key: str | None = None,
) -> str:
    v3 = model == "dall-e-3"
    client = openai.AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    response = await client.images.generate(
        prompt=prompt,
        model=model,
        response_format=response_format,
        quality="hd" if hd and v3 else NOT_GIVEN,
        size=size,
        style=style if v3 else NOT_GIVEN,
    )
    image_result = response.data[0].url or response.data[0].b64_json
    assert image_result is not None
    return image_result


async def edit(
    image: str | Path,
    prompt: str,
    mask: str | Path | None = None,
    model: Literal["dall-e-2"] = "dall-e-2",
    response_format: Literal["url", "b64_json"] = "url",
    size: Literal["256x256", "512x512", "1024x1024"] = "1024x1024",
    api_key: str | None = None,
):
    client = openai.AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    with open(image, "rb") as image_file:
        if mask is not None:
            with open(mask, "rb") as mask_file:
                response = await client.images.edit(
                    image=image_file,
                    prompt=prompt,
                    mask=mask_file,
                    model=model,
                    response_format=response_format,
                    size=size,
                )
        else:
            response = await client.images.edit(
                image=image_file,
                prompt=prompt,
                model=model,
                response_format=response_format,
                size=size,
            )
    image_result = response.data[0].url or response.data[0].b64_json
    assert image_result is not None
    return image_result


async def create_variation(
    image: str | Path,
    model: Literal["dall-e-2"] = "dall-e-2",
    response_format: Literal["url", "b64_json"] = "url",
    size: Literal["256x256", "512x512", "1024x1024"] = "1024x1024",
    api_key: str | None = None,
):
    client = openai.AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    with open(image, "rb") as image_file:
        response = await client.images.create_variation(
            image=image_file,
            model=model,
            response_format=response_format,
            size=size,
        )
    image_result = response.data[0].url or response.data[0].b64_json
    assert image_result is not None
    return image_result


async def vision(image: str | Path, prompt: str, api_key: str | None = None) -> str:
    # Covert image to base64
    if isinstance(image, str) and (
        image.startswith("http://")
        or image.startswith("https://")
        or image.startswith("data:image/")
    ):
        image_url = image
    else:
        ext = f"{image}".split(".")[-1]
        with open(image, "rb") as f:
            image_url = (
                f"data:image/{ext};base64,{base64.b64encode(f.read()).decode('utf-8')}"
            )
    gpt = AugmentedGPT(
        model="gpt-4-vision-preview",
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
    )
    response = gpt.chat_completion(
        [
            Message(
                role=Role.USER,
                content=[
                    ContentPartText(prompt),
                    ContentPartImage(image_url),
                ],
            ),
        ],
    )
    async for msg in response:
        result = msg.content
        assert isinstance(result, str)
        return result

    return "No response from the model."
