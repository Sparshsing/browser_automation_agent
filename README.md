# Browser Automation Agent

## [Crustdata Build Challenge : AI Agent for Browser Automation]

This project is a browser automation agent that can understand natural language instructions and perform actions in a web browser, using Gemini LLMs and Playwright.

## Architecture

The agent consists of three main components:

1. **Planner**: Breaks down a user's natural language query into high-level steps.
2. **Executor**: Performs each step by interacting with the browser through Playwright.
3. **Verifier**: Verifies if each step was completed successfully.

## Setup

1. Clone the repository
2. Install the requirements:
   ```
   pip install -r requirements.txt
   ```
3. Install Playwright browsers:
   ```
   playwright install chromium
   ```
4. Create a `.env` file with your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

Run the main script:
```
python main.py
```

Enter your natural language query when prompted. For example:
```
login to reddit.com, search ghibli, save the first three posts with metadata like date, author etc.
```

The agent will:
1. Plan the steps needed to achieve your goal
2. Execute each step in the browser
3. Verify each step was completed successfully
4. Report progress and results

## Configuration

Edit `config.py` to modify the agent's behavior:
- Change model names
- Enable/disable headless mode
- Adjust retry limits
- Enable/disable human-in-the-loop mode

## Files

- `main.py`: Entry point script
- `models.py`: Data models
- `planner.py`: Step planning module
- `executor.py`: Browser interaction module
- `verifier.py`: Step verification module
- `utils.py`: Utility functions
- `config.py`: Configuration settings 