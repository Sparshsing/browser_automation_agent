import io
import os
import json
import base64
import logging
import asyncio
import validators
import time
import sqlite3
from datetime import datetime
from collections import defaultdict
from PIL import Image
from bs4 import BeautifulSoup
from playwright.async_api import Page, Browser
from google.genai import types
import config


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

# Initialize SQLite database
def init_db():
    """
    Initialize SQLite database with required tables
    """
    if not os.path.exists(os.path.dirname(config.DB_PATH)):
        os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    
    # Create comprehensive gemini_calls table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS gemini_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            input_tokens INTEGER,
            output_tokens INTEGER,
            total_tokens INTEGER,
            user_request_json TEXT,
            response_json TEXT,
            chat_history_json TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def store_gemini_call(input_tokens, output_tokens, total_tokens, user_request_json, response_json, chat_history_json):
    """
    Store Gemini API call details in database
    """
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        """INSERT INTO gemini_calls 
           (input_tokens, output_tokens, total_tokens, user_request_json, response_json, chat_history_json)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (input_tokens, output_tokens, total_tokens, 
         json.dumps(user_request_json), json.dumps(response_json), json.dumps(chat_history_json))
    )
    
    conn.commit()
    conn.close()

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
                if part.inline_data is not None:
                    continue  # skip previous images
                if part.text and len(part.text) > 0:
                    part_trimmed = part.model_copy()
                    if len(part_trimmed.text) > 500:
                        part_trimmed.text = part_trimmed.text[:250] + '... [Trimmed] ...' + part_trimmed.text[-250:]
                    parts_to_add.append(part_trimmed)
                else:
                    parts_to_add.append(part.model_copy())
            user_content = types.UserContent(parts=parts_to_add)
            trimmed_chat_history.append(user_content)
        else:
            trimmed_chat_history.append(content.model_copy())
    return trimmed_chat_history


def get_chat_history_json(history):
    trimmed_history = []
    
    for message in history:
        # Create new message with trimmed parts
        trimmed_parts = []
        
        for part in message.parts:
            # Handle text content
            if part.text is not None:
                part_trimmed = part.model_copy()
                if len(part_trimmed.text) > 500:
                    part_trimmed.text = part_trimmed.text[:250] + f'... [Trimmed {len(part_trimmed.text)-500} characters] ...' + part_trimmed.text[-250:]
                trimmed_parts.append(part_trimmed)
            
            # Handle file data
            elif part.file_data is not None:
                part_trimmed = part.model_copy()
                if len(str(part_trimmed.file_data)) > 100:
                    part_trimmed.file_data = str(part_trimmed.file_data)[:50] + f'... [Trimmed {len(str(part_trimmed.file_data))-100} characters] ...' + str(part_trimmed.file_data)[-50:]
                trimmed_parts.append(part_trimmed)

            # Handle inline data
            elif part.inline_data is not None:
                part_trimmed = part.model_copy()
                if len(str(part_trimmed.inline_data)) > 100:
                    part_trimmed.inline_data = str(part_trimmed.inline_data)[:50] + f'... [Trimmed {len(str(part_trimmed.inline_data))-500} characters] ...' + str(part_trimmed.inline_data)[-50:]
                trimmed_parts.append(part_trimmed)
            
            # Keep other part types as-is
            else:
                trimmed_parts.append(part)
        
        # Create new message with trimmed parts
        if message.role == "user":
            trimmed_message = message.model_copy()
            trimmed_message.parts = trimmed_parts
        else:
            trimmed_message = message.model_copy()
            trimmed_message.parts = trimmed_parts
            
        trimmed_history.append(trimmed_message)
    
    trimmed_history_json = [message.to_json_dict() for message in trimmed_history]
    return trimmed_history_json



def redact_passwords_in_logs(log_dir, password):

    if not password or not log_dir:
        return
        
    # Walk through directory
    for root, _, files in os.walk(log_dir):
        for filename in files:
            if filename.endswith('.log'):
                filepath = os.path.join(root, filename)
                
                # Read file content
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Replace exact password matches with asterisks
                modified = content.replace(password, '*' * len(password))
                
                # Write back if modified
                if modified != content:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(modified)



# Track request timestamps for rate limiting
_request_timestamps = defaultdict(list)
# ADDED: Track tokens used for rate limiting
_token_usage = defaultdict(list)

def call_gemini_chat(chat, content, max_retries=2):
    """
    Call the chat with given content and log metrics, including any tool calls.
    Respects rate limits and includes retry logic.
    
    Args:
        chat: The chat instance to use
        content: The content/prompt to send
        rate_limits: Dictionary mapping model names to requests per minute limits
        max_retries: Maximum number of retry attempts on error (default: 2)
    
    Returns:
        The chat response
    """
    global _request_timestamps
    model_name = getattr(chat, '_model', 'default')
    
    # Rate limiting logic for request count
    rate_limits = config.RATE_LIMITS
    if rate_limits and model_name in rate_limits:
        rpm_limit = rate_limits[model_name]
        current_time = time.time()
        
        # Remove timestamps older than 60 seconds
        _request_timestamps[model_name] = [ts for ts in _request_timestamps[model_name] if current_time - ts < 60]
        
        # Check if we're over the limit
        if len(_request_timestamps[model_name]) >= rpm_limit:
            oldest_timestamp = min(_request_timestamps[model_name])
            wait_time = 60 - (current_time - oldest_timestamp) + 5  # buffer time
            if wait_time > 0:
                logger.info(f"Rate limit reached for {model_name}. Waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
    
    # Record this request timestamp
    _request_timestamps[model_name].append(time.time())
    
    # ADDED: Token usage rate limiting
    token_limit = config.TOKEN_LIMITS.get(model_name, float('inf'))
    global _token_usage
    current_time = time.time()
    _token_usage[model_name] = [(ts, tkn) for ts, tkn in _token_usage[model_name] if current_time - ts < 60]
    total_recent_tokens = sum(tkn for ts, tkn in _token_usage[model_name])
    while total_recent_tokens >= token_limit:
        oldest_timestamp = min(ts for ts, tkn in _token_usage[model_name])
        wait_time = 60 - (current_time - oldest_timestamp) + 5
        logger.info(f"Token usage limit reached for {model_name}. Waiting {wait_time:.2f} seconds")
        time.sleep(wait_time)
        current_time = time.time()
        _token_usage[model_name] = [(ts, tkn) for ts, tkn in _token_usage[model_name] if current_time - ts < 60]
        total_recent_tokens = sum(tkn for ts, tkn in _token_usage[model_name])
    
    # Retry logic
    retry_count = 0
    while retry_count <= max_retries:
        try:
            # Get chat history JSON
            chat_history_json = {}  #get_chat_history_json(chat.get_history())

            # Prepare user request JSON
            user_request_json =  ([types.UserContent(parts=content).to_json_dict()])
            
            # Send message and get response
            logger.info(f"Sending message to LLM: {user_request_json}")
            response = chat.send_message(content)
            
            # Get response JSON
            response_json = get_chat_history_json([response.candidates[0].content])
            
            # Extract metrics
            usage = response.usage_metadata
            prompt_tokens = usage.prompt_token_count
            output_tokens = usage.candidates_token_count
            total_tokens = prompt_tokens + output_tokens
            logger.info(f"Token usage: {prompt_tokens} prompt, {output_tokens} output, {total_tokens} total")
            
            # ADDED: Update token usage tracking with tokens from this call
            _token_usage[model_name].append((time.time(), total_tokens))
            
            # Store all information in database
            store_gemini_call(
                input_tokens=prompt_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                user_request_json=user_request_json,
                response_json=response_json,
                chat_history_json=chat_history_json
            )
            
            # Get prompt and response text for logging
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
            
        except Exception as e:
            retry_count += 1
            if retry_count <= max_retries:
                logger.warning(f"Error calling Gemini API: {str(e)}. Retrying ({retry_count}/{max_retries})...")
                time.sleep(60)  # Wait a second before retrying
            else:
                logger.error(f"Failed to call Gemini API after {max_retries} retries: {str(e)}")
                raise 
            

