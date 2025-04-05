from google.genai import types
import config
from models import Step
from utils import call_gemini_chat, create_custom_logger

logfile = "logs/agent.log"
logger = create_custom_logger(__name__, logfile)

def create_planner_chat(client):
    """
    Creates a planner chat session with appropriate configurations
    
    Args:
        client: The Genai client to use
    
    Returns:
        A configured chat session for the planner
    """
    planner_instructions = """
    **Role:** You are an expert AI planning agent specializing in breaking down user requests into high-level, sequential steps for web browser automation.

    **Task:** Analyze the user's natural language query and decompose it into a logical sequence of distinct, high-level goals or sub-tasks that need to be performed within a web browser to fulfill the request.

    **Input:** You will be given a user query describing a task they want to automate in a web browser.

    **Output Requirements:**
    Generate a JSON object containing a list of `"steps"`, where each "step" object represents one high-level step in the automation plan.
    """
    
    planner_config = types.GenerateContentConfig(
        system_instruction=planner_instructions,
        temperature=0.1,
        response_mime_type='application/json',
        response_schema=list[Step]
    )
    
    return client.chats.create(model=config.PLANNING_MODEL, config=planner_config)


def plan_user_query(client, user_query):
    """
    Processes a user query and breaks it down into steps
    
    Args:
        client: The Genai client to use
        user_query: The user's natural language query
    
    Returns:
        A list of Steps or None if planning fails
    """
    logger.info(f"Planning user query: {user_query}")
    
    try:
        # Create planner chat
        planner_chat = create_planner_chat(client)
        
        # Format user query
        user_message = f"\n\nUser Query: {user_query}"
        
        # Get steps from planner
        response = call_gemini_chat(planner_chat, user_message)
        
        # Parse steps from response
        steps = response.parsed
        
        if not steps or len(steps) < 1:
            logger.error("No steps returned from planner")
            return None
        
        logger.info(f"Planned {len(steps)} steps for query: {user_query}")
        for step in steps:
            logger.info(f"Step {step.step_id}: {step.goal}")
            
        return steps
    
    except Exception as e:
        logger.error(f"Error in planning: {e}")
        return None 