from datetime import datetime


def name_timestamp(format:str='%d%m%Y_%H%M%S') -> str:
    '''Returns a timestamp string using a
    given timestamp format. Defaults to the current
    time in day,month,year_hours,minutes,seconds format.
    '''
    return datetime.now().strftime(format)