from datetime import datetime


def name_timestamp(format='%d%m%Y_%H%M%S'):
    return datetime.now().strftime(format)