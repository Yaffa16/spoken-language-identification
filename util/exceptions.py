class NotLoadedError(Exception):
    """Exception for not loaded files when trying to perform some operation"""
    def __init__(self, message):
        super(Exception, self).__init__(message)
