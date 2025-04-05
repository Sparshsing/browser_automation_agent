import asyncio
import validators
from bs4 import BeautifulSoup
from playwright.async_api import Page, Browser, TimeoutError
from google.genai import types

import config
from utils import (
    call_gemini_chat, get_simplified_dom, get_trimmed_chat_history,
    create_custom_logger
)

logfile = "logs/agent.log"
logger = create_custom_logger(__name__, logfile)

# Define function declarations for tool calling
goto_url_declaration = {
    "name": "goto_url",
    "description": "Navigate the current page to the specified URL.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to navigate to."
            }
        },
        "required": ["url"]
    }
}

open_new_page_declaration = {
    "name": "open_new_page",
    "description": "Opens a new page in the current browser context and navigates to the specified URL.",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to navigate the new page to."
            }
        },
        "required": ["url"]
    }
}

perform_locator_action_declaration = {
    "name": "perform_locator_action",
    "description": "Performs a specified action on a page element located by a selector.",
    "parameters": {
        "type": "object",
        "properties": {
            "selector": {
                "type": "string",
                "description": "A string containing a CSS or XPath selector to locate the element."
            },
            "action_name": {
                "type": "string",
                "description": "Name of the Playwright Locator method to call on the element."
            },
            "args_dict": {
                "type": "object",
                "description": "Dictionary of arguments to pass to the Locator method."
            }
        },
        "required": ["selector", "action_name", "args_dict"]
    }
}

perform_page_action_declaration = {
    "name": "perform_page_action",
    "description": "Performs a specified action directly on the current page object.",
    "parameters": {
        "type": "object",
        "properties": {
            "action_name": {
                "type": "string",
                "description": "Name of the Playwright Page method to call."
            },
            "args_dict": {
                "type": "object",
                "description": "Dictionary of arguments to pass to the Page method."
            }
        },
        "required": ["action_name", "args_dict"]
    }
}

get_user_input_declaration = {
    "name": "get_user_input",
    "description": "Ask for user input based on a given query",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The prompt/question to display to the user"
            }
        },
        "required": ["query"]
    }
}

def create_executor_chat(client):
    """
    Creates an executor chat session with appropriate configurations
    
    Args:
        client: The Genai client to use
    
    Returns:
        A configured chat session for the executor
    """
    executor_system_instruction = """
    You are an expert Playwright automation agent. Your goal is to achieve a specific task ('Step Goal') within a web browser environment using the provided tools.

    **Process:**
    1.  You will be given a 'Step Goal' representing one part of a larger user request.
    2.  You will receive the current page URL and a 'simplified_dom' JSON representing the interactive elements currently visible on the page.
    3.  Analyze the 'Step Goal' and the current page state ('simplified_dom', URL).
    4.  Use the available tools (`goto_url`, `open_new_page`, `perform_locator_action`, `perform_page_action`) by making function calls to interact with the browser page.
    5.  Ask the user for any information/confirmation needed (eg. credentials, a confirmation, etc), in case you do not have enough information to proceed.
    6.  You may need to make **multiple sequential function calls** to fully achieve the 'Step Goal'. For example, you might need to 'fill' an input and then 'click' a button.
    7.  After each function call you make, you will receive a 'FunctionResponse' indicating the outcome (success message, data returned, or error message).
    8.  **Use the 'FunctionResponse'**: If an action failed, analyze the error and try a different approach (e.g., a different selector, a different action). If an action succeeded, proceed with the next necessary action.
    9.  **Selector Strategy**: Aim for robust selectors (IDs, unique attributes like 'data-testid' or 'name', ARIA labels, text content). Use Playwright syntax (e.g., `input[name='username']`, `button:has-text('Log In')`). Refer to the 'simplified_dom' for available attributes and text.
    10.  **Completion**: Once you believe the specific 'Step Goal' provided has been fully achieved based on the sequence of actions and their results, **STOP making function calls** and respond with a short text message confirming completion (e.g., "Step completed: Logged into Quora successfully.") or indicating failure if the goal cannot be achieved after reasonable attempts (e.g., "Step failed: Could not find the search input after trying multiple selectors.").

    **Important:** Only focus on the *current* 'Step Goal'. Do not attempt actions related to subsequent steps in the overall plan. Be precise and methodical.
    """
    
    # Create tool object with all function declarations
    tools = types.Tool(function_declarations=[
        goto_url_declaration,
        open_new_page_declaration,
        perform_locator_action_declaration,
        get_user_input_declaration
        # perform_page_action_declaration  # Uncomment if needed
    ])
    
    # Create executor config
    executor_config = types.GenerateContentConfig(
        system_instruction=executor_system_instruction,
        temperature=0.1,
        tools=[tools]
    )
    
    return client.chats.create(model=config.EXECUTION_MODEL, config=executor_config)


def create_step_prompt(step_goal, current_url, simplified_dom, function_response_parts, step_logs, action_failure):
    """
    Creates the prompt parts for executing a single step
    """
    failure_fix_prompt = "An action has failed. Please try again. You may decide to change the plan or action based on current state, to achieve the goal."
    
    prompt_text = f"""
        Current Step Goal:
        ---
        {step_goal}
        ---

        Current Page State:
        URL: {current_url}
        Simplified DOM:
        {simplified_dom}

        ---

        The logs of previous actions done to achieve the goal:
        {step_logs if len(step_logs) > 0 else ''}
        Consider the previous actions that have already been completed for the current goal and perform the next actions necessary.

        ---
        Use your tools sequentially to achieve the Current Step Goal based on the page state.
        **Important:** Remember to provide a final text response when the goal is achieved or definitively fails, instead of making another function call.
    """

    if action_failure:
        prompt_text = prompt_text + f"\n\n{failure_fix_prompt}"

    executor_prompt = function_response_parts + [types.Part.from_text(text=prompt_text)]
    return executor_prompt


async def goto_url(url: str, active_page: Page) -> None:
    """
    Navigate the current page to the specified URL.
    """
    await active_page.goto(url, wait_until="domcontentloaded", timeout=30000)


async def open_new_page(url: str, browser: Browser) -> Page:
    """
    Opens a new page in the current browser context and navigates to the specified URL.
    """
    new_page = await browser.contexts[0].new_page()
    await new_page.goto(url, wait_until="domcontentloaded", timeout=30000)
    return new_page


async def perform_locator_action(selector: str, action_name: str, args_dict: dict, active_page: Page) -> None:
    """
    Performs a specified action on a page element located by a selector.
    """
    locator = active_page.locator(selector)
    # Call the function dynamically
    if hasattr(locator, action_name):
        method = getattr(locator, action_name)
        await method(**args_dict)  # Pass arguments if required
    else:
        raise AttributeError(f"Invalid action name: {action_name}")


async def perform_page_action(action_name: str, args_dict: dict, active_page: Page) -> None:
    """
    Performs a specified action directly on the current page object.
    """
    # Call the function dynamically
    if hasattr(active_page, action_name):
        method = getattr(active_page, action_name)
        # Check if the method is awaitable
        if asyncio.iscoroutinefunction(method):
            await method(**args_dict)  # Pass arguments if required
        else:
            method(**args_dict) # Call non-async methods directly
    else:
        raise AttributeError(f"Invalid page action name: {action_name}")


def get_user_input(query: str) -> str:
    """
    Prompts the user for input based on a given query.
    """
    return input(query)


async def execute_step(client, current_step_id, current_step_goal, active_page, browser):
    """
    Executes a single step of the plan on the browser
    
    Args:
        client: The Genai client
        current_step_id: The id of the current step
        current_step_goal: The goal of the current step
        active_page: The active Playwright page
        browser: The Playwright browser instance
    
    Returns:
        A tuple containing (success, active_page, final_text, step_logs)
    """
    max_retries = config.MAX_RETRIES
    max_consecutive_tool_calls = config.MAX_CONSECUTIVE_TOOL_CALLS
    retry_count = 0
    num_tool_calls = 0
    step_completed_successfully = False
    action_failure = False
    function_response_parts = []
    step_logs = []
    
    logger.info(f"Executing step {current_step_id}: {current_step_goal}")
    
    # Create executor chat
    executor_chat = create_executor_chat(client)
    
    # Get initial DOM and URL
    current_url = active_page.url
    current_url_valid = validators.url(current_url)
    simplified_dom = None
    
    if current_url_valid:
        simplified_dom = await get_simplified_dom(active_page)
        simplified_dom = BeautifulSoup(simplified_dom, 'html.parser').prettify()
    
    # Executor loop
    while num_tool_calls < max_consecutive_tool_calls and retry_count < max_retries and not step_completed_successfully:
        # Update DOM if URL has changed
        if active_page.url != current_url:
            current_url = active_page.url
            current_url_valid = validators.url(current_url)
            if current_url_valid:
                simplified_dom = await get_simplified_dom(active_page)
                simplified_dom = BeautifulSoup(simplified_dom, 'html.parser').prettify()
                step_logs.append(f"DOM updated for the current url: {current_url}")
            else:
                simplified_dom = None
        
        # Check for failures in previous iteration
        if action_failure:
            step_logs.append(f"Attempt {retry_count+1}, current_url: {current_url}\n")
            logger.info(f"Attempt {retry_count+1}, current_url: {current_url}\n")
        
        logger.info(f"'goal' {current_step_goal}, 'current_url', {current_url}, 'simplified_dom is None?' {simplified_dom is None}")
        
        # Trim chat history for efficiency
        executor_chat_history = executor_chat.get_history()
        if len(executor_chat_history) > 0:
            trimmed_chat_history = get_trimmed_chat_history(executor_chat_history)
            executor_chat = client.chats.create(
                model=config.EXECUTION_MODEL, 
                config=executor_chat._config, 
                history=trimmed_chat_history
            )
        
        # Construct and send prompt to LLM
        executor_prompt = create_step_prompt(
            current_step_goal, 
            current_url, 
            simplified_dom, 
            function_response_parts, 
            step_logs, 
            action_failure
        )
        
        try:
            response = call_gemini_chat(executor_chat, executor_prompt)
            logger.info(f"Called Step Executor model. Response text: {response.text}")
            step_logs.append(f"Called Step Executor model. Response text: {response.text}")
        except Exception as e:
            logger.exception(f"Error calling executor model: {e}")
            return False, active_page, f"Error: {e}", step_logs
        
        # Check if this is a final response (no function calls)
        if not response.function_calls or len(response.function_calls) == 0:
            final_text = response.text
            logger.info(f"Step Executor provided final text for Step {current_step_id}: {final_text}")
            step_logs.append(f"Step Executor provided final text for Step {current_step_id}: {final_text}")
            
            # Basic check if LLM indicated success
            if "complete" in final_text.lower() or "success" in final_text.lower() or "achieved" in final_text.lower():
                step_completed_successfully = True
            else:
                logger.warning(f"Warning: Step Executor finished step but message doesn't clearly indicate success: '{final_text}'")
                step_completed_successfully = False  # Be conservative
            
            break  # Exit the loop
        
        # Process function calls
        step_logs.append(f"Function Calls: {len(response.function_calls)} calls made")
        action_failure = False
        function_response_parts = []
        
        for function_call in response.function_calls:
            num_tool_calls += 1
            
            function_name = function_call.name
            args = dict(function_call.args)
            logger.info(f"LLM requested Function Call: {function_name}({args})")
            step_logs.append(f"LLM requested Function Call: {function_name}({args})")
            
            # Execute the requested function
            function_result = None
            
            try:
                if function_name == "goto_url":
                    await goto_url(url=args["url"], active_page=active_page)
                    logger.info(f"URL updated to: {active_page.url}")
                    step_logs.append(f"URL updated to: {active_page.url}")
                    function_result = "Action completed successfully"
                    await asyncio.sleep(1)
                
                elif function_name == "open_new_page":
                    new_page = await open_new_page(url=args["url"], browser=browser)
                    active_page = new_page
                    function_result = "Action completed successfully"
                    logger.info(f"Switched active page to: {active_page.url}")
                    step_logs.append(f"Switched active page to: {active_page.url}")
                    await asyncio.sleep(1)
                
                elif function_name == "perform_locator_action":
                    await perform_locator_action(
                        selector=args["selector"],
                        action_name=args["action_name"],
                        args_dict=args["args_dict"],
                        active_page=active_page
                    )
                    logger.info(f"Completed locator action: {function_name}({args})")
                    step_logs.append(f"Completed locator action: {function_name}({args})")
                    function_result = "Action completed successfully"
                    await asyncio.sleep(1)
                
                elif function_name == "perform_page_action":
                    await perform_page_action(
                        action_name=args["action_name"],
                        args_dict=args["args_dict"],
                        active_page=active_page
                    )
                    logger.info(f"Completed page action: {function_name}({args})")
                    step_logs.append(f"Completed page action: {function_name}({args})")
                    function_result = "Action completed successfully"
                    await asyncio.sleep(1)
                
                elif function_name == "get_user_input":
                    function_result = get_user_input(query=args["query"])
                    logger.info(f"Completed get_user_input: {function_name}({args})")
                    step_logs.append(f"Completed get_user_input: {function_name}({args})")
                
                else:
                    logger.error(f"Unknown function call requested: {function_name}")
                    function_result = f"Error: Unknown function: {function_name}"
                    action_failure = True
            
            except Exception as e:
                function_result = f"Error Completing Action: {e}"
                logger.exception(f"Error executing {function_name}: {e}")
                action_failure = True
            
            # Create function response part
            if function_result is None:
                function_result = "Action completed successfully"
            
            logger.info(f"Function Result: {function_result}")
            step_logs.append(f"Function Result: {function_result}")
            
            function_response_part = types.Part.from_function_response(
                name=function_call.name,
                response={"result": function_result},
            )
            function_response_parts.append(function_response_part)
            
            # Delay to allow browser to update
            await asyncio.sleep(0.5)
            
            # Break loop if action failed to update retry count
            if action_failure:
                retry_count += 1
                break
            
            # Update DOM and URL if needed
            if active_page.url != current_url:
                current_url = active_page.url
                current_url_valid = validators.url(current_url)
                if current_url_valid:
                    simplified_dom = await get_simplified_dom(active_page)
                    simplified_dom = BeautifulSoup(simplified_dom, 'html.parser').prettify()
                else:
                    simplified_dom = None
    
    # Final status check
    if not step_completed_successfully:
        logger.warning(f"Step {current_step_id} FAILED after {retry_count} retries or {num_tool_calls} tool calls")
        return False, active_page, f"Failed to complete step {current_step_id}: {current_step_goal}", step_logs
    else:
        logger.info(f"Step {current_step_id}-{current_step_goal} COMPLETED successfully")
        return True, active_page, final_text, step_logs