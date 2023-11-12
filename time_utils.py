
from datetime import datetime

class timeMeasure:
    def __init__(self):
        None

    def start(self):
        self.startTime = datetime.now()


    def end_new(self, info=''):
        self.endTime = datetime.now()
        t = self.endTime - self.startTime
        # print 't:',t
        if t.seconds > 60:
            print(info, int(t.seconds / float(60)), ' mins. ', int(t.seconds % 60), 'secs.')
        elif t.seconds <= 60 and t.seconds > 0:
            print(info, t.seconds, ' secs. ', int(t.microseconds / float(10 ** 3)), 'msecs.')
        elif t.seconds == 0:
            print(info, int(t.microseconds / float(10 ** 3)), ' msecs. ', int(t.microseconds % float(10 ** 3)), 'usecs.')

    def end(self, info=''):
        self.endTime = datetime.now()
        t = self.endTime - self.startTime
        # print 't:',t
        if t.seconds > 60:
            print (info, t.seconds / float(60), ' mins. ')
        # elif t.seconds <= 60 and t.seconds > 0:
        elif 60 >= t.seconds > 0:
            print(info,"%.2f secs." % (t.seconds + t.microseconds / float(10 ** 6)))
            # t.seconds+t.microseconds/float(10**6),' secs. ',

        elif t.seconds == 0 and t.microseconds > 100:
            print(info, "%.2f msecs." % (t.microseconds / float(10 ** 3)))
        elif t.seconds == 0 and t.microseconds < 100:
            print (info, t.microseconds, 'usecs')

def get_hour_minutes():
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%H%M")

    return current_time