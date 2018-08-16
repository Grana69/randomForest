import csv

from random import randrange
from datetime import timedelta

from datetime import datetime
import random

def random_date(start, end):
    """
    This function will return a random datetime between two datetime 
    objects.
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return start + timedelta(seconds=random_second)

def generate_randomHour(d1,d2):
        randomHour = random_date(d1, d2)
        return convertToHours(randomHour)
    
def convertToHours(z1):
    rHour = ''
    if(z1.minute<10):
        rHour += str(z1.hour) + '0' + str(z1.minute)
    else:
        rHour += str(z1.hour) + str(z1.minute)
    return rHour

with open('nostis.csv', mode='w') as csv_file:
    fieldnames = ['haveBusy', 'busyStart', 'busyEnd','day','timeSent','timeAnswered','answered']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    d1 = datetime.strptime('1/1/2009 6:00 AM', '%m/%d/%Y %I:%M %p')
    d2 = datetime.strptime('1/1/2009 10:00 PM', '%m/%d/%Y %I:%M %p')

    bsH1 = datetime.strptime('1/1/2009 6:00 AM', '%m/%d/%Y %I:%M %p')#start
    bsH2 = datetime.strptime('1/1/2009 10:00 AM', '%m/%d/%Y %I:%M %p')
    beH1 = datetime.strptime('1/1/2009 4:00 PM', '%m/%d/%Y %I:%M %p')
    beH2 = datetime.strptime('1/1/2009 10:00 PM', '%m/%d/%Y %I:%M %p')#end
    zeroTime = 0
    for x in range(0,500000):
        r1 = random_date(bsH1,beH2)
        r2 = random_date(r1,beH2)
        if(random.randint(0, 1)):
            if(random.randint(0, 1)):
                writer.writerow({'haveBusy': '1', 'busyStart': generate_randomHour(bsH1,bsH2), 'busyEnd': generate_randomHour(beH1,beH2),'day': random.randint(1, 7),'timeSent': convertToHours(r1),'timeAnswered': convertToHours(r2),'answered': 1})
            else:
                writer.writerow({'haveBusy': '1', 'busyStart': generate_randomHour(bsH1,bsH2), 'busyEnd': generate_randomHour(beH1,beH2),'day': random.randint(1, 7),'timeSent': zeroTime,'timeAnswered': zeroTime,'answered': 0})
        else:
            if(random.randint(0, 1)):
                writer.writerow({'haveBusy': '0', 'busyStart': zeroTime, 'busyEnd': zeroTime,'day': random.randint(1, 7),'timeSent': convertToHours(r1),'timeAnswered': convertToHours(r2),'answered': 1})
            else:
                writer.writerow({'haveBusy': '0', 'busyStart': zeroTime, 'busyEnd': zeroTime,'day': random.randint(1, 7),'timeSent': zeroTime,'timeAnswered': zeroTime,'answered': 0})