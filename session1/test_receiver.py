#from camera import camera
from alphabot2.ultrasonic import ultrasonic
from alphabot2.infrared import infrared
from alphabot2.line_tracker import line_tracker

import time
address='192.168.0.100'

inf = infrared(address)
inf.start()
ult = ultrasonic(address)
ult.start()
lt = line_tracker(address)
lt.start()


while True:
	try:
		
		print("infrared left,right: ",inf.left,inf.right)
		print("ultrasonic: ",ult.measurement)
		print("line_tracker: ",lt.data)
		time.sleep(1)
			
	except KeyboardInterrupt:

		inf.stop()
		ult.stop()
		lt.stop()

		break




