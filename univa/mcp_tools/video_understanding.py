import os
import yaml

from univa.mcp_tools.base import ToolResponse, setup_logger
from univa.utils.query_llm import multimodal_query



# Load configuration
# config_path = "config/mcp_tools_config/config.yaml"
# os.chdir(os.path.dirname(os.path.dirname(__file__)))
config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config/mcp_tools_config")
config_path = os.path.join(config_dir, "config.yaml")
if not os.path.exists(config_path):
    config_path = os.path.join(config_dir, "config.example.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

video_understanding_config = config.get('video_understanding', {})

# Configure logging
logger = setup_logger(__name__, "logs/mcp_tools", "video_understanding.log")
logger.info(f"Loaded video_understanding_config: {video_understanding_config}")

# Tool function (direct, no MCP server)
def vision2text_gen(prompt: str, multimodal_path: str, type: str) -> dict:
    """
    Analyzes and describes the content of a video or image based on a given prompt, converting visual information into text.
    This tool is useful for understanding ambiguous or complex visual inputs, providing detailed textual descriptions of the content.

    Args:
        prompt (str): User's instruction.
        multimodal_path (str): The path of the video or image.
        type (str): The type of the multimodal input, either "video" or "image".

    Returns:
        dict: A dictionary containing the success status and a message.
              - 'success' (bool): True if the vision content was understood successfully, False otherwise.
              - 'message' (str, optional): The details of the vision content if successful.
              - 'error' (str, optional): An error message if the operation failed.
    """
    try:
        # This tool routes to an LLM-based multimodal endpoint; it does not require torch.
        if type == "video":
            content = multimodal_query(prompt, video_path=multimodal_path)
        elif type == "image":
            content = multimodal_query(prompt, image_path=multimodal_path)
        else:
            return ToolResponse(
                success=False,
                message="The type of the multimodal input should be either 'video' or 'image'."
            )

        return ToolResponse(
            success=True,
            message="Vision content understood successfully.",
            content=content
        )
    except Exception as e:
        return ToolResponse(
            success=False,
            message=f"An error occurred: {str(e)}"
        )
