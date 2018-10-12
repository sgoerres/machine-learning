import datetime

# initialize list of timers with generic overall start time
timers = {"____" : datetime.datetime.now()}

# return runtime for specified timer in seconds
def _rt(timer="____"): # runtime
    return (datetime.datetime.now() - timers[timer]).total_seconds()

# start new timer
def _s(timer):
     timers.update({timer: datetime.datetime.now()})
