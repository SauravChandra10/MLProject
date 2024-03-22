import sys
from flask import jsonify
# from src.mlproject.logger import logging

def error_message_detail(error,error_detail:sys):
    """
    This function extracts details about a Python exception for better error reporting.

    Args:
        error: The exception object.
        error_detail: The sys.exc_info() tuple containing exception details.

    Returns:
        A formatted string containing the error message, filename, and line number.
    """
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
     file_name,exc_tb.tb_lineno,str(error))

    return error_message


class CustomException(Exception):
    """
    A custom exception class to handle errors within the application.
    """
    def __init__(self,error_message,error_details:sys):
        """
        Initialize the CustomException with the original error message and enhanced details.

        Args:
            error_message: The original error message.
            error_details: The sys.exc_info() tuple containing exception details.
        """
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_details)

    def __str__(self):
        """
        Returns the formatted error message with details for better debugging.
        """
        return self.error_message