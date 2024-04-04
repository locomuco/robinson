from __future__ import division
import time
import board
from adafruit_motor import servo
from adafruit_pca9685 import PCA9685

i2c = board.I2C()
pca = PCA9685(i2c)
pca.frequency = 60

servo_eyelid_top_right = servo.Servo(pca.channels[5])
servo_eyelid_bottom_right = servo.Servo(pca.channels[4])
servo_eyelid_top_left = servo.Servo(pca.channels[3])
servo_eyelid_bottom_left = servo.Servo(pca.channels[2])
servo_up_down = servo.Servo(pca.channels[1])
servo_left_right = servo.Servo(pca.channels[0])


def close_right_eye():
    servo_eyelid_top_right.angle = 0
    servo_eyelid_bottom_right.angle = 90


def open_right_eye():
    servo_eyelid_top_right.angle = 50
    servo_eyelid_bottom_right.angle = 0


def open_left_eye():
    servo_eyelid_top_left.angle = 0
    servo_eyelid_bottom_left.angle = 90


def close_left_eye():
    servo_eyelid_top_left.angle = 50
    servo_eyelid_bottom_left.angle = 20


def look_right():
    servo_left_right.angle = 30


def look_left():
    servo_left_right.angle = 140


def look_up():
    servo_up_down.angle = 90


def look_down():
    servo_up_down.angle = 0


def blink(blink_time):
    close_left_eye()
    close_right_eye()
    time.sleep(blink_time)
    open_left_eye()
    open_right_eye()


if __name__ == "__main__":
    blink(0.08)
    time.sleep(0.4)
    look_left()
    blink(0.08)
    time.sleep(0.6)
    look_right()
    time.sleep(0.7)
    look_left()
    time.sleep(0.7)
