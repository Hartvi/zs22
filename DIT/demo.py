#!/usr/bin/python

import guide
import time

def callback(msg):
    print("callback: "+msg)

g = guide.Guide()
g.connect("/dev/ttyUSB0")

g.buttonCallback = callback
g.moveCallback = callback
g.homeCallback = callback

print(g.id())

print("going home")
g.home()
while not g.isHome():
  time.sleep(3)
  print("still not home")

g.set(guide.LEDG, 1)
time.sleep(0.5)
g.set(guide.LEDY, 1)
time.sleep(0.5)
g.set(guide.LEDR, 1)
time.sleep(3)
g.set(guide.LEDS, 0)
time.sleep(0.5)


print("going far away")
small_unit = 100
to_meters = 1e-3
dist = 0
dist /= to_meters
# g.moveTo(100)
start_time = time.time()
while True:
    print("dist:", dist*to_meters, "m")
    print("press enter to continue or enter 'n' to stop")
    x = input()
    if "n" in x:
        print("EXITING")
        break
    else:
        dist += small_unit
    g.moveTo(dist)
    last_position = 0
    while g.get(guide.POSITION) != last_position:
        last_position = g.get(guide.POSITION)
    # print("time taken to reach", dist, ": ", time.time()-start_time)

# for i in range(1000):
    # time.sleep(3)
    # print("position: "+str(g.get(guide.POSITION)), end="")

# for i in range(10):
#     time.sleep(3)
#     print("position: "+str(g.get(guide.POSITION)))
# g.moveTo(10)

print("disarm and quit")
g.disarm()
g.quit()

