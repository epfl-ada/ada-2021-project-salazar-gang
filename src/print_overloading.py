from __future__ import print_function
import builtins
import io
import time
import datetime

global LOGS_FILE
LOGS_FILE = None
global LOGS_TIME
LOGS_TIME = None
global LOGS_BUFFER
LOGS_BUFFER = None

def overload_with_save(func=print):
    '''
    This function modify the built-in print function.
    The modified print version save the print outputs in a text file (logs.txt)
    :param func: MUST be func=print
    :return: overloaded print function which save the print outputs in a text file (logs.txt)
    '''
    def old_function(func):
        def wrapped_func(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapped_func
    print_old = old_function(print)
    def wrapped_func(*args,**kwargs):
        global LOGS_FILE, LOGS_TIME, LOGS_BUFFER
        out = io.StringIO()
        print_old(*args, **kwargs, file=out)
        timestamp = str(datetime.datetime.now())[:19] + ":\t"
        newvalue = timestamp + out.getvalue()
        newvalue = newvalue.replace("\n", f"\n{timestamp}", newvalue.count("\n")-1)
        LOGS_BUFFER = LOGS_BUFFER + newvalue
        # if time.time() - LOGS_TIME > 1.0:
        #     LOGS_TIME += 1.0
        with open(f"{LOGS_FILE}", mode='a') as f:
            f.write(LOGS_BUFFER)
            LOGS_BUFFER = ""
        return func(*args,**kwargs)
    return wrapped_func

def save_print():
    '''
    Initialize the savings of print statements.
    After calling this functions, all print will be saved inside a log.txt file.
    '''
    global LOGS_FILE, LOGS_TIME, LOGS_BUFFER
    timestamp = str(datetime.datetime.now()).replace(' ', '_').replace(':', 'h')[:16]
    LOGS_FILE = f"logs_{timestamp}.txt"
    LOGS_TIME = time.time()
    LOGS_BUFFER = ""
    builtins.print = overload_with_save(print)