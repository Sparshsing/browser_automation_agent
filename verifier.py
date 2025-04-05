from google.genai import types
from bs4 import BeautifulSoup
import validators

import config
from models import VerificationResult
from utils import (
    call_gemini_chat, get_page_screenshot, get_simplified_dom,
    create_custom_logger
)

logfile = "logs/agent.log"
logger = create_custom_logger(__name__, logfile)

def create_verifier_chat(client):
    """
    Creates a verifier chat session with appropriate configurations
    
    Args:
        client: The Genai client to use
    
    Returns:
        A configured chat session for the verifier
    """
    verifier_system_instruction = """
    You are a helpful assistant that verifies if a particular goal related to a browser session has been completed successfully.
    You will be given a goal, a list of steps that have been taken by another agent, and a final text response from an LLM. 
    You will also be provided with a screenshot and DOM of the current page.
    You will need to determine if the goal has been completed successfully.
    """
    
    verifier_config = types.GenerateContentConfig(
        system_instruction=verifier_system_instruction,
        response_mime_type='application/json',
        response_schema=VerificationResult,
        temperature=0.1
    )
    
    return client.chats.create(model=config.VERIFIER_MODEL, config=verifier_config)


async def verify_step_completion(client, step, active_page, step_logs, final_text, current_step_goal):
    """
    Verifies if a step has been completed successfully
    
    Args:
        client: The Genai client
        step: The Step object that was executed
        active_page: The active Playwright page
        step_logs: The logs from the step execution
        final_text: The final text from the executor
        current_step_goal: current updated goal for this step
    
    Returns:
        A VerificationResult object
    """
    logger.info(f"Verifying step completion for step {step.step_id}: {step.goal}")
    
    # Get current page state
    current_url = active_page.url
    current_url_valid = validators.url(current_url)
    simplified_dom = None
    active_page_screenshot = None
    
    try:
        # Get DOM and screenshot
        if current_url_valid:
            simplified_dom = await get_simplified_dom(active_page)
            simplified_dom = BeautifulSoup(simplified_dom, 'html.parser').prettify()
            active_page_screenshot = await get_page_screenshot(active_page, full_page=True)
        else:
            verifier_issue = f'Invalid URL {active_page.url}. Cannot proceed with verification'
            verification_result = VerificationResult(success=False, message=verifier_issue, new_goal=current_step_goal)
            return verification_result
    except Exception as e:
        logger.exception(f"Error getting page state for verification: {e}")
        raise e
    
    # Prepare verification prompt
    verifier_prompt = f"""
    Current Step Goal:
    {step.goal}

    Current Page State:
    URL: {current_url}
    Simplified DOM:
    {simplified_dom}

    Steps Taken:
    {step_logs}

    Final Text Response from Execution Agent:
    {final_text}

    Please verify if the Goal has been completed, based on the screenshot, dom, url, step logs and the final text response. See if there are any errors being shown on the page.

    If the goal has been completed, provide a success message.

    If the goal has not been completed, 
        1. provide a modified goal that should be completed, based on the current state (from screenshot, dom, url, step logs). Ultimately to achieve the goal.
        2. mention the issue identified or any other message that can help the execution agent to achieve the goal.

    Output a valid JSON object.
    """
    
    # Create verifier chat
    verifier_chat = create_verifier_chat(client)
    
    try:
        # Call verifier with screenshot and text
        verifier_message = [active_page_screenshot, types.Part.from_text(text=verifier_prompt)]
        verifier_response = call_gemini_chat(verifier_chat, verifier_message)
        
        # Parse verification result
        verification_result = verifier_response.parsed
        
        logger.info(f"Verification result: success={verification_result.success}, message={verification_result.message}")
        
        return verification_result
    
    except Exception as e:
        logger.exception(f"Error in verification: {e}")
        raise e
        # Create a default failure result
        # return VerificationResult(
        #     success=False, 
        #     message=f"Error during verification: {e}",
        #     new_goal=current_step_goal  # Keep the same goal to retry
        # ) 