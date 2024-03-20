import sys
# from src.logger import logging
from logger import logging


def error_message_detail(error, error_detail:sys):
    # in this exc_tb will have all the details of the error
    # like in which file error occured, line number and error message
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
    file_name, exc_tb.tb_lineno, str(error))

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        self.error_message = error_message_detail(error_message, error_detail)
        logging.error(self.error_message)
        # logging.error("Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
        #     file_name, exc_tb.tb_lineno, str(error)))

    def __str__(self):
        return self.error_message 
    
 
    
    
if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as e:
        logging.info("Error occured in python script ") 
        raise CustomException(e, sys)
    
        

    

