import logging
import os
from datetime import datetime 

# Define the log file name dynamically based on current date and time
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Construct the log file path by joining current working directory, "logs" folder, and the log file name
log_path = os.path.join(os.getcwd(),"logs",LOG_FILE)

# Create the "logs" directory if it doesn't exist (avoiding errors)
os.makedirs(log_path,exist_ok=True)

# Combine the path to the "logs" directory and the log file name for full path
LOG_FILE_PATH = os.path.join(log_path,LOG_FILE)

# Configure basic logging with the following settings:
# - Log file: Use the constructed LOG_FILE_PATH
# - Format: Include timestamp, line number, logger name, log level, and message
# - Level: Set log level to INFO (will capture informational and higher severity messages)
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)