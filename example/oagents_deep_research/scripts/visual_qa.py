import base64
import json
import mimetypes
import os
import uuid
from io import BytesIO
from typing import Optional

import requests
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from PIL import Image
from transformers import AutoProcessor

from smolagents import Tool, tool


load_dotenv(override=True)

idefics_processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b-chatty")

# Tracker for HF InferenceClient (idefics2-8b-chatty)
idefics_hf_tracker = {
    "model_id": "HuggingFaceM4/idefics2-8b-chatty",
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_tokens": 0,
    "api_calls": 0,
}

def reset_idefics_hf_tracker():
    global idefics_hf_tracker
    idefics_hf_tracker = {
        "model_id": "HuggingFaceM4/idefics2-8b-chatty",
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "api_calls": 0,
    }

def get_cumulative_idefics_hf_details() -> dict:
    return idefics_hf_tracker.copy()

# Tracker for the direct gpt-4o call in visualizer function
visualizer_gpt4o_tracker = {
    "model_id": "gpt-4o-2024-11-20", # As specified in payload
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_tokens": 0,
    "total_input_cost": 0.0,
    "total_output_cost": 0.0,
    "total_cost": 0.0,
    "api_calls": 0,
}

def reset_visualizer_gpt4o_tracker():
    global visualizer_gpt4o_tracker
    visualizer_gpt4o_tracker = {
        "model_id": "gpt-4o-2024-11-20",
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "total_input_cost": 0.0,
        "total_output_cost": 0.0,
        "total_cost": 0.0,
        "api_calls": 0,
    }

def get_cumulative_visualizer_gpt4o_details() -> dict:
    return visualizer_gpt4o_tracker.copy()

def process_images_and_text(image_path, query, client):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": query},
            ],
        },
    ]

    prompt_with_template = idefics_processor.apply_chat_template(messages, add_generation_prompt=True)

    # load images from local directory

    # encode images to strings which can be sent to the endpoint
    def encode_local_image(image_path):
        # load image
        image = Image.open(image_path).convert("RGB")

        # Convert the image to a base64 string
        buffer = BytesIO()
        image.save(buffer, format="JPEG")  # Use the appropriate format (e.g., JPEG, PNG)
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # add string formatting required by the endpoint
        image_string = f"data:image/jpeg;base64,{base64_image}"

        return image_string

    image_string = encode_local_image(image_path)
    prompt_with_images = prompt_with_template.replace("<image>", "![]({}) ").format(image_string)

    payload = {
        "inputs": prompt_with_images,
        "parameters": {
            "return_full_text": False,
            "max_new_tokens": 200,
        },
    }

    raw_response = client.post(json=payload)
    response_data = json.loads(raw_response.decode())[0]

    # Attempt to get token counts - THIS IS SPECULATIVE for HF Inference API
    # The actual way to get token counts depends on the specific model server's response format.
    # HF text-generation-inference might provide `details` with `prefill` (input) and `generated_tokens`.
    prompt_tokens = 0
    completion_tokens = 0
    if response_data and isinstance(response_data, dict) and "details" in response_data:
        if "prefill" in response_data["details"] and isinstance(response_data["details"]["prefill"], list) and len(response_data["details"]["prefill"]) > 0:
            # Assuming prefill tokens are in the first element's 'tokens' field if it exists
            if isinstance(response_data["details"]["prefill"][0], dict) and "tokens" in response_data["details"]["prefill"][0]:
                 prompt_tokens = response_data["details"]["prefill"][0]["tokens"]
        # 'generated_tokens' is often directly available
        completion_tokens = response_data["details"].get("generated_tokens", 0)
    elif response_data and isinstance(response_data, dict) and "generated_text" in response_data and "inputs" in payload:
        # Fallback: estimate based on string length if no direct token count (very rough)
        # This is not a good way to count tokens and should be replaced if possible.
        # prompt_tokens = len(payload["inputs"]) // 4 # Rough estimate
        # completion_tokens = len(response_data["generated_text"]) // 4 # Rough estimate
        logger.warning("HF InferenceClient token count not directly available in response, relying on generated_tokens or falling back to 0.")
        # If only generated_tokens is available from details, use that. Otherwise, might need to use tokenizer manually if precise counts are needed.
        if "details" in response_data and "generated_tokens" in response_data["details"]:
            completion_tokens = response_data["details"].get("generated_tokens",0)
        else: # Unable to determine tokens
            prompt_tokens = 0
            completion_tokens = 0
            logger.warning("Cannot determine prompt or completion tokens for HF idefics call.")

    global idefics_hf_tracker
    idefics_hf_tracker["total_prompt_tokens"] += prompt_tokens
    idefics_hf_tracker["total_completion_tokens"] += completion_tokens
    idefics_hf_tracker["total_tokens"] += (prompt_tokens + completion_tokens)
    idefics_hf_tracker["api_calls"] += 1
    logger.debug(f"HF Idefics Call - Prompt Tokens: {prompt_tokens}, Completion Tokens: {completion_tokens}")
    logger.debug(f"HF Idefics Cumulative - Total Calls: {idefics_hf_tracker['api_calls']}, Total Tokens: {idefics_hf_tracker['total_tokens']}")

    return response_data.get("generated_text", "") # Return the generated text


# Function to encode the image
def encode_image(image_path):
    if image_path.startswith("http"):
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
        request_kwargs = {
            "headers": {"User-Agent": user_agent},
            "stream": True,
        }

        # Send a HTTP request to the URL
        response = requests.get(image_path, **request_kwargs)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")

        extension = mimetypes.guess_extension(content_type)
        if extension is None:
            extension = ".download"

        fname = str(uuid.uuid4()) + extension
        download_path = os.path.abspath(os.path.join("downloads", fname))

        with open(download_path, "wb") as fh:
            for chunk in response.iter_content(chunk_size=512):
                fh.write(chunk)

        image_path = download_path

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}


def resize_image(image_path):
    img = Image.open(image_path)
    width, height = img.size
    img = img.resize((int(width / 2), int(height / 2)))
    new_image_path = f"resized_{image_path}"
    img.save(new_image_path)
    return new_image_path


class VisualQATool(Tool):
    name = "visualizer"
    description = "A tool that can answer questions about attached images."
    inputs = {
        "image_path": {
            "description": "The path to the image on which to answer the question",
            "type": "string",
        },
        "question": {"description": "the question to answer", "type": "string", "nullable": True},
    }
    output_type = "string"

    client = InferenceClient("HuggingFaceM4/idefics2-8b-chatty")

    def forward(self, image_path: str, question: Optional[str] = None) -> str:
        output = ""
        add_note = False
        if not question:
            add_note = True
            question = "Please write a detailed caption for this image."
        try:
            output = process_images_and_text(image_path, question, self.client)
        except Exception as e:
            print(e)
            if "Payload Too Large" in str(e):
                new_image_path = resize_image(image_path)
                output = process_images_and_text(new_image_path, question, self.client)

        if add_note:
            output = (
                f"You did not provide a particular question, so here is a detailed caption for the image: {output}"
            )

        return output


@tool
def visualizer(image_path: str, question: Optional[str] = None) -> str:
    """A tool that can answer questions about attached images.

    Args:
        image_path: The path to the image on which to answer the question. This should be a local path to downloaded image.
        question: The question to answer.
    """

    add_note = False
    if not question:
        add_note = True
        question = "Please write a detailed caption for this image."
    if not isinstance(image_path, str):
        raise Exception("You should provide at least `image_path` string argument to this tool!")

    mime_type, _ = mimetypes.guess_type(image_path)
    base64_image = encode_image(image_path)

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                ],
            }
        ],
        "max_tokens": 1000,
    }
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status() # Raise an exception for HTTP errors
        response_json = response.json()
        output = response_json["choices"][0]["message"]["content"]

        # Cost tracking for gpt-4o
        prompt_tokens = 0
        completion_tokens = 0
        if "usage" in response_json:
            prompt_tokens = response_json["usage"].get("prompt_tokens", 0)
            completion_tokens = response_json["usage"].get("completion_tokens", 0)
        else:
            logger.warning("Usage field not found in visualizer gpt-4o response. Cannot track tokens.")

        current_input_cost, current_output_cost, current_total_cost = calculate_cost(
            payload["model"], prompt_tokens, completion_tokens, is_embedding=False
        )

        global visualizer_gpt4o_tracker
        visualizer_gpt4o_tracker["total_prompt_tokens"] += prompt_tokens
        visualizer_gpt4o_tracker["total_completion_tokens"] += completion_tokens
        visualizer_gpt4o_tracker["total_tokens"] += (prompt_tokens + completion_tokens)
        visualizer_gpt4o_tracker["total_input_cost"] += current_input_cost
        visualizer_gpt4o_tracker["total_output_cost"] += current_output_cost
        visualizer_gpt4o_tracker["total_cost"] += current_total_cost
        visualizer_gpt4o_tracker["api_calls"] += 1
        
        logger.debug(
            f"Visualizer GPT-4o Call - Prompt Tokens: {prompt_tokens}, Completion Tokens: {completion_tokens}, "
            f"Input Cost: ${current_input_cost:.6f}, Output Cost: ${current_output_cost:.6f}, Call Total Cost: ${current_total_cost:.6f}"
        )
        logger.debug(
            f"Visualizer GPT-4o Cumulative - Total Calls: {visualizer_gpt4o_tracker['api_calls']}, Total Tokens: {visualizer_gpt4o_tracker['total_tokens']}, Total Cost: ${visualizer_gpt4o_tracker['total_cost']:.6f}"
        )

    except requests.exceptions.RequestException as e:
        logger.error(f"Visualizer HTTP Request failed: {e}")
        output="Failed to get response from visualizer API."
    except (KeyError, IndexError, TypeError) as e:
        # raise Exception(f"Response format unexpected: {response.json()}")
        logger.error(f"Visualizer API response format error: {e}. Response: {response.text if 'response' in locals() else 'N/A'}")
        output="None valid question due to unsolvable problem, please set your final answer to Unable to determine."

    if add_note:
        output = f"You did not provide a particular question, so here is a detailed caption for the image: {output}"

    return output
