#!/usr/bin/python

import serial
import threading
import sys, tty, termios
import time


class port:
    def __init__(self, port):
        self.ComPort = serial.Serial(port) # open COM24
        self.ComPort.baudrate = 115200 # set Baud rate to 115200
        self.ComPort.bytesize = 8    # Number of data bits = 8
        self.ComPort.parity   = 'N'  # No parity
        self.ComPort.stopbits = 1    # Number of Stop bits = 1
        self.ComPort.timeout=3
        self.ComPort.flushInput()
        self.run = True
        self.handler = None
        self.thread = threading.Thread(target=self.read_from_port, args=())
        self.thread.start()

    def read_from_port(self):
        while self.run:
            if (self.ComPort.inWaiting() > 0):
              data_str = self.ComPort.readline().decode('ascii')
              if (not data_str is None):
                  if (not self.handler is None):
                      self.handler(data_str)
            time.sleep(0.01)

    def stop(self):
        self.run=False
        self.thread.join()
        self.ComPort.close()

    def write(self, ch):
        line = ch+"\r"
        data = bytearray(line, 'ascii')
        No = self.ComPort.write(data)


#port names   i dont want to write qvotes around it
LEDR = "LEDR"
LEDG = "LEDG"
LEDY = "LEDY"
LEDS = "LEDS"

OC0 = "OC0"
OC0 = "OC1"
OC0 = "OC2"

DIOE = "DIOE"
DO0 = "DO0"
DO1 = "DO1"
DI0 = "DO0"
DI1 = "DO1"

REF = "REF"
AO0 = "AO0"
AO1 = "AO1"
AI0 = "AO0"
AI1 = "AO1"

BTN = "BTN"
SCALE = "SCALE"
TRESHOLD = "CTRESH"
POWER_V = "UIN"
IS_HOME = "HOME"
POSITION = "POS"
ACTUAL_CURRENT = "CUR"
MAXIMAL_CURRENT = "CURM"

class Guide:
    def __init__(self):
        self.messages = []
        self.maxMessages = 64
        self.messageEvent  = threading.Event()
        self.timeout = 5
        self.buttonCallback = None
        self.moveCallback = None
        self.homeCallback = None

    def connect(self, prt):
        self.com = port(prt)
        self.com.handler = self.messageHandler

    def messageHandler(self, msg):
        if msg.startswith("MOVE_E:"):
            if not self.moveCallback is None:
                self.moveCallback(msg)
        elif msg.startswith("HOME_E:"):
            if not self.homeCallback is None:
                self.homeCallback(msg)
        elif msg.startswith("BTN_E:"):
            if not self.buttonCallback is None:
                self.buttonCallback(msg)
        self.messages.append(msg)
        self.messageEvent.set()
        self.messageEvent.clear()
        if len(self.messages) > self.maxMessages:
          self.messages.pop()

    def waitOnMessage(self, msg):
        for i in range(self.timeout*10):
          self.messageEvent.wait(0.1)
          for x in self.messages:
            if x.startswith(msg):
              self.messages.remove(x)
              return x
        
        for x in self.messages:
          print(x)
        raise Exception('Comunication timeout! Waiting on:' + msg)

    def quit(self):
        self.disarm()
        self.com.stop()

    def checkAnswer(self, msg):
        s = self.waitOnMessage(msg+":")
        if not s.startswith(msg+":OK"):
            raise Exception('Execution error "'+s+'"')

    def execute(self, cmd):
        self.com.write(cmd+"!")
        self.checkAnswer(cmd)


# COMMANDS

    def id(self):
        self.com.write("ID?")
        return self.waitOnMessage("ID:")[3:]

    def reset(self):
        self.com.write("RST!")
        self.waitOnMessage("ID:")

    def set(self, pin, value):
        if value is True:
          value = 1
        if value is False:
          value = 0
        line = pin+"!"+str(value)
        self.com.write(line)
        self.checkAnswer(pin)

    def get(self, pin):
        line = pin+"?"
        self.com.write(line)
        s = self.waitOnMessage(pin+":").replace(pin+":","")
        return float(s)

    def getD(self, pin):
        return self.get(pin) > 0

    def home(self):
        self.execute("HOME")

    def disarm(self):
        self.execute("DISARM")

    def moveTo(self, pos, speed = 15):
        line = "MOVE!"+str(pos)+";"+str(speed)
        self.com.write(line)
        self.checkAnswer("MOVE")

    def getPosition(self, pos, speed = 15):
        return self.get(POSITION)

    def setOrigin(self):
        self.execute("ZERO")

    def isHome(self):
        return self.getD(IS_HOME)

    def resetMaxCurrent(self):
        self.execute("DIS")






