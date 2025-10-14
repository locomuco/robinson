from __future__ import division
import time
import board
import random
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


def blink(blink_time=0.08):
    close_left_eye()
    close_right_eye()
    time.sleep(blink_time)
    open_left_eye()
    open_right_eye()
    # A quick post-blink pause to make it look more natural
    time.sleep(0.05)

# --- New Control Functions ---

def center_gaze():
    """Sets a neutral, forward-looking gaze."""
    servo_left_right.angle = 85 # Center point between 30 (right) and 140 (left)
    servo_up_down.angle = 45 # Center point between 0 (down) and 90 (up)

def randomized_gaze(center_lr=85, center_ud=45):
    """Shifts the gaze to a new, random, but controlled position."""
    # Randomly select a new angle for left/right and up/down
    # Max range: LR (30 to 140) and UD (0 to 90)
    # Use center points and a small offset for subtle shifts
    new_lr = random.randint(center_lr - 30, center_lr + 30)
    new_ud = random.randint(center_ud - 20, center_ud + 20)

    # Ensure angles stay within the safe/defined range
    new_lr = max(30, min(140, new_lr))
    new_ud = max(0, min(90, new_ud))

    # Move the servos
    servo_left_right.angle = new_lr
    servo_up_down.angle = new_ud

def natural_gaze():
    """Continuously moves the eyes with blinks for a natural look."""
    print("Starting natural gaze loop. Press Ctrl+C to stop.")
    center_gaze()
    open_right_eye()
    open_left_eye()

    try:
        while True:
            # 1. Random Blink (Occurs about 1 in 10 cycles for a natural rate)
            if random.random() < 0.5:
                blink(random.uniform(0.06, 0.1)) # Variable blink time

            # 2. Shift Gaze
            randomized_gaze()

            # 3. Hold Gaze for a random period (mimicking human focus)
            gaze_time = random.uniform(0.5, 3.0)
            time.sleep(gaze_time)

            # 4. Return to Center/Subtle Movement (Slight "micro-shift" before the next gaze)
            center_gaze()
            time.sleep(random.uniform(0.1, 0.3))

    except KeyboardInterrupt:
        print("\nGaze loop stopped. Centering eyes...")
        center_gaze()
        # Optional: Add a final blink for a complete stop
        blink(0.1)
        sys.exit()

# --- Main Execution ---

if __name__ == "__main__":
    # Remove the old scripted sequence and call the new continuous function
    # blink(0.08)
    # time.sleep(0.4)
    # look_left()
    # blink(0.4)
    # time.sleep(0.6)
    # look_right()
    # time.sleep(0.7)
    # look_left()
    # time.sleep(0.7)

    natural_gaze()
