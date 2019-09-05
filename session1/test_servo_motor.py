# from servo import servo
from alphabot2.motor import motor
import time

address = "192.168.0.100"

mot=motor(address)
# mot.command("forward",10,2)
# mot.command("backward",10,2)
# mot.command("left",10,2)
# mot.command("right",10,2)

# mot.command("set_left_speed",50)
# mot.command("set_right_speed",50)
# mot.command("forward")

mot.command("backward")
time.sleep(5)

mot.stop()