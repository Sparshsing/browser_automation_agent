import os
import asyncio
from dotenv import load_dotenv

from google import genai
from playwright.async_api import async_playwright

import logging
import config
from models import Step
from planner import plan_user_query
from executor import execute_step
from verifier import verify_step_completion
from utils import create_custom_logger, init_db

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="google.*")

logfile = "logs/agent.log"
logger = create_custom_logger(__name__, logfile)


# suppress google warnings
logging.getLogger('google_genai.types').setLevel(logging.ERROR)

# set log level as Warning for Root logger
logging.basicConfig(level=logging.WARNING)

load_dotenv()

async def interact(user_query: str):
    """
    Browser Interaction Agent
    """
    # Initialize database to log llm calls and responses
    init_db()
    
    # Initialize Google Genai client
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    
    
    
    # Plan the query into steps
    print("Making a plan for the query...")
    steps = plan_user_query(client, user_query)
    
    if not steps:
        print("Failed to plan steps for the query.")
        return
    
    print(f"Planned {len(steps)} steps:")
    logger.info(f"Planned {len(steps)} steps:")
    for step in steps:
        print(f"Step {step.step_id}: {step.goal}")
        logger.info(f"Step {step.step_id}: {step.goal}")
    
    # Start browser
    async with async_playwright() as playwright:
        
        try:
            # Launch browser
            browser = await playwright.chromium.launch(headless=config.HEADLESS)

            # Create initial page
            page = await browser.new_page()
            active_page = page
            
            # Execute each step of the plan
            for step_index, step in enumerate(steps):
                print(f"\nExecuting step {step.step_id} of {len(steps)}: {step.goal}")
                logger.info(f"\nExecuting step {step.step_id} of {len(steps)}: {step.goal}")
                
                # Try to execute the step
                step_success = False
                verifier_message = None
                step_retry_count = 0
                max_step_retries = 2  # Max retries for a failed step with verification
                current_step_goal = step.goal

                while not step_success and step_retry_count < max_step_retries:
                    if step_retry_count > 0:
                        print(f"Retrying step {step.step_id} (attempt {step_retry_count + 1})")
                        logger.info(f"Retrying step {step.step_id} (attempt {step_retry_count + 1})")
                    
                    # Execute step
                    try:
                        success = False
                        final_text = ""
                        success, active_page, final_text, step_logs = await execute_step(
                            client, step.step_id, current_step_goal, active_page, browser, verifier_message=verifier_message
                        )
                        if not success and final_text == f"User Aborted the program":
                            exit()
                    except Exception as e:
                        logger.exception(f"Error during step execution: {e}")
                        print(f"Error during step execution: {e}")
                        step_retry_count += 1
                        continue
                    
                    if success:
                        # Verify step completion
                        verification = await verify_step_completion(
                            client, step, active_page, step_logs, final_text, current_step_goal=current_step_goal
                        )
                        
                        if verification.success:
                            step_success = True
                            print(f"Step {step.step_id} completed successfully: {verification.message}")
                            logger.info(f"Step {step.step_id} completed successfully: {verification.message}")
                            print(f"Response from LLM: \n{final_text}")
                        else:
                            print(f"Step execution verified as FAILED: {verification.message}")
                            logger.info(f"Step execution verified as FAILED: {verification.message}")
                            if verification.new_goal:
                                print(f"Updated goal (due to partial execution) for step {step.step_id}: {verification.new_goal}")
                                # Update step goal for retry
                                current_step_goal = verification.new_goal
                    else:
                        print(f"Step {step.step_id} execution failed: {final_text}")
                        logger.info(f"Step {step.step_id} execution failed: {final_text}")
                    step_retry_count += 1
                
                # If step failed afte r all retries, stop execution
                if not step_success:
                    print(f"Failed to complete step {step.step_id} after {step_retry_count} attempts. Stopping.")
                    logger.info(f"Failed to complete step {step.step_id} after {step_retry_count} attempts. Stopping.")
                    break
                
                # If this was the last step, report success
                if step_index == len(steps) - 1:
                    print("\nAll steps completed successfully!")
                    logger.info("\nAll steps completed successfully!")
        except Exception as e:
            logger.exception(f"Error during execution: {e}")
            print(f"Error during execution: {e}")
         
        finally:
            # Close browser
            await browser.close()
            print("\nBrowser closed.")


if __name__ == "__main__":
    # Run the main function
    # Get user query
    # user_query = input("Enter your query for browser automation: ")
    user_query = "open bbc.com and login, search for rcb ipl and click the second link. provide a 100 word summary and include any metadata"
    logger.info(f"User query: {user_query}")
    asyncio.run(interact(user_query)) 