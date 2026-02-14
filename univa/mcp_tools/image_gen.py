import base64
import yaml
import os
from datetime import datetime

from univa.mcp_tools.base import ToolResponse, setup_logger
from univa.utils.image_process import download_image
from univa.utils.wavespeed_api import text_to_image_generate, seedream_v4_edit, seedream_v4_sequential_edit

# Load configuration
# config_path = "config/mcp_tools_config/config.yaml"
# os.chdir(os.path.dirname(os.path.dirname(__file__)))
config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config/mcp_tools_config")
config_path = os.path.join(config_dir, "config.yaml")
if not os.path.exists(config_path):
    config_path = os.path.join(config_dir, "config.example.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

image_gen_config = config.get('image_gen', {})

def _get_wavespeed_api_key() -> str:
    """API keys must come from environment variables (.env)."""
    key = os.getenv("WAVESPEED_API_KEY") or ""
    if not key:
        raise RuntimeError("Missing WAVESPEED_API_KEY (set it in .env or environment).")
    return key

# Configure logging
logger = setup_logger(__name__)



def text2image_generate(prompt: str)-> ToolResponse:
    """
    Generates a new image based on a textual prompt.
    This tool is useful for creating visual content from scratch.

    Args:
        prompt (str): The text prompt to use for image generation.

    Returns:
        dict: A dictionary containing the success status, output image path, and a message.
              - 'success' (bool): True if the image was generated successfully, False otherwise.
              - 'output_image' (str, optional): The path to the generated image if successful.
              - 'message' (str, optional): A success message.
              - 'error' (str, optional): An error message if the generation failed.
    """
    model = image_gen_config.get("text_to_image")

    if model == "flux-kontext":
        api_key = _get_wavespeed_api_key()
        image_url = text_to_image_generate(api_key, prompt)
        _time = datetime.now().strftime("%m%d%H%M%S")
        base_output_path = image_gen_config.get("base_output_path", "results/image")
        os.makedirs(base_output_path, exist_ok=True)
        image_save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), f"{base_output_path}/{_time}_{prompt[:30].replace(' ', '_')}.jpg")

    logger.info(f"Image URL: {image_url}")
    download_image(image_url, save_path=image_save_path)
    logger.info(f"Image saved to: {image_save_path}")

    return ToolResponse(
        success=True,
        output_path=image_save_path,
        message="Image generated successfully."
    )




def image2image_generate(prompt: str, image_path: str|list[str]):
    """
    Generates a new image based on a text prompt and an input image or a list of images, while maintaining consistency with characters or styles from the reference images.
    This tool is useful for modifying existing images or generating new ones with specific visual references.

    Args:
        prompt (str): The text prompt describing the desired image modifications or style.
        image_path (str | list[str]): The file path to the initial image to be used as a base for generation. Can be a single path or a list of paths.

    Returns:
        dict: A dictionary containing the success status, output image path, and a message.
              - 'success' (bool): True if the image was generated successfully, False otherwise.
              - 'output_image' (str, optional): The path to the generated image if successful.
              - 'message' (str, optional): A success message.
              - 'error' (str, optional): An error message if the generation failed.
    """

    model = image_gen_config.get("image_to_image")
    
    if model == "flux-kontext":
        api_key = _get_wavespeed_api_key()
        res = seedream_v4_edit(api_key, prompt, image_path)
        image_url = res["output_path"]
        _time = datetime.now().strftime("%m%d%H%M%S")
        base_output_path = image_gen_config.get("base_output_path", "results/image")
        os.makedirs(base_output_path, exist_ok=True)
        image_save_path = f"{base_output_path}/{_time}_{prompt[:30].replace(' ', '_')}.jpg"

    logger.info(f"Image URL: {image_url}")
    download_image(image_url, save_path=image_save_path)
    logger.info(f"Image saved to: {image_save_path}")

    return ToolResponse(
        success=True,
        output_path=image_save_path,
        message="Image generated successfully."
    )


def sequential_image_gen(prompt: str, images: list[str], images_num: int = 2) -> str:
    """
    Generates a series of same style or consistency images or based on input images and a prompt.
    This tool is useful for creating multiple related images based on input images and a prompt.

    Args:
        prompt (str): The text prompt describing the desired image modifications or style.
        images (list[str]): List of image file paths to be used as base for generation (up to 10 images).
        images_num (int): Number of images to generate, must align with the number of input images.

    Returns:
        dict: A dictionary containing the success status, output URLs, and a message.
              - 'success' (bool): True if the images were generated successfully, False otherwise.
              - 'output_urls' (list, optional): List of URLs to the generated images if successful.
              - 'message' (str, optional): A success message.
              - 'error' (str, optional): An error message if the generation failed.
    """
    api_key = _get_wavespeed_api_key()
    
    try:
        result = seedream_v4_sequential_edit(
            api_key=api_key,
            prompt=prompt,
            images=images,
            max_images=images_num
        )

        output_images = result.get('output_path')
        output_paths = []
        for item in output_images:
            _time = datetime.now().strftime("%m%d%H%M%S")
            base_output_path = image_gen_config.get("base_output_path", "results/image")
            os.makedirs(base_output_path, exist_ok=True)
            image_save_path = f"{base_output_path}/{_time}_{prompt[:30].replace(' ', '_')}.jpg"
            download_image(item, save_path=image_save_path)
            output_paths.append(image_save_path)
        
        if result.get('success'):
            return ToolResponse(
                success=True,
                output_path=output_paths,
                message="Sequential image editing completed successfully."
            )
        else:
            return ToolResponse(
                success=False,
                error=result.get('error', 'Unknown error occurred during sequential image editing.')
            )
    except Exception as e:
        return ToolResponse(
            success=False,
            error=f"Error during sequential image editing: {str(e)}"
        )

