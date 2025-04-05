import io
import os
import json
import base64
import logging
import asyncio
import validators
from PIL import Image
from bs4 import BeautifulSoup
from playwright.async_api import Page, Browser
import crawl4ai.content_scraping_strategy as scrapper
from google.genai import types

# Setup logging
def create_custom_logger(logger_name, logfile_path):
    """
    Creates a custom logger with file handler
    """
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    
    if not os.path.exists(os.path.dirname(logfile_path)):
        os.makedirs(os.path.dirname(logfile_path), exist_ok=True)
    
    handler = logging.FileHandler(logfile_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger

# Initialize loggers
logfile = "logs/agent.log"
logger = create_custom_logger(__name__, logfile)

llm_logfile = "logs/llm.log"
llm_logger = create_custom_logger("llm_logger", llm_logfile)

async def get_page_screenshot(page, full_page=True):
    """
    Get a screenshot of the current page as a PIL Image
    """
    screenshot_bytes = await page.screenshot(full_page=full_page)
    pil_image = Image.open(io.BytesIO(screenshot_bytes))
    return pil_image

def image_to_base64(image):
    """
    Converts a PIL Image object to a Base64 encoded string
    """
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_byte = buffered.getvalue()
    img_str = base64.b64encode(img_byte).decode('utf-8')
    return img_str

async def get_simplified_dom(page):
    """
    Get a simplified DOM representation of the current page
    """
    html_content = await page.content()
    scrapping_strategy = scrapper.WebScrapingStrategy()
    scrap_result = scrapping_strategy.scrap(url=page.url, html=html_content)
    simplified_dom = scrap_result.cleaned_html
    return simplified_dom

def get_token_count(client, content, model):
    """
    Get the token count for a given content and model
    """
    return client.models.count_tokens(model=model, contents=content).total_tokens

def get_trimmed_content(content):
    """
    Trims content to avoid excessively long content
    """
    content_str = str(content)
    if len(content_str) > 500:
        return content_str[:250] + f' ... [Trimmed {len(content_str)-500} characters] ... ' + content_str[-250:]
    else:
        return content_str

def get_trimmed_chat_history(history):
    """
    Trims chat history to avoid excessively long messages
    """
    trimmed_chat_history = []
    for content in history:
        if content.role == 'user':
            parts_to_add = []
            for part in content.parts:
                if part.text and len(part.text) > 0:
                    part_trimmed = part.copy()
                    part_trimmed.text = part_trimmed.text[:100] + '... [Trimmed]'
                    parts_to_add.append(part_trimmed)
                else:
                    parts_to_add.append(part.model_copy())
            user_content = types.UserContent(parts=parts_to_add)
            trimmed_chat_history.append(user_content)
        else:
            trimmed_chat_history.append(content.model_copy())
    return trimmed_chat_history

def call_gemini_chat(chat, content):
    """
    Call the chat with given content and log metrics, including any tool calls
    
    Args:
        chat: The chat instance to use
        content: The content/prompt to send
    
    Returns:
        The chat response
    """
    response = chat.send_message(content)
    
    # Extract metrics
    usage = response.usage_metadata
    prompt_tokens = usage.prompt_token_count
    output_tokens = usage.candidates_token_count
    total_tokens = prompt_tokens + output_tokens
    
    # Get prompt and response text
    if hasattr(response, 'text'):
        output_text = response.text
    else:
        output_text = str(response)
    
    # Check for tool calls
    tool_call_info = None
    tool_call_tokens = 0
    
    if hasattr(response, 'function_calls') and response.function_calls:
        tool_call_info = []
        for function_call in response.function_calls:
            tool_call_info.append({
                'name': function_call.name,
                'args': dict(function_call.args)
            })

    # Prepare log data
    log_data = {
        'prompt_tokens': prompt_tokens,
        'output_tokens': output_tokens,
        'total_tokens': total_tokens,
        'prompt': get_trimmed_content(content),
        'response': output_text
    }
    
    # Add tool call info if present
    if tool_call_info:
        log_data['tool_call'] = tool_call_info
        if tool_call_tokens:
            log_data['tool_call_tokens'] = tool_call_tokens
    
    # Log everything
    llm_logger.info(log_data)
    
    return response 