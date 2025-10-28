import cv2
import mediapipe as mp
import pyautogui
import time
import threading
import math

# Initializing video capture
capture = cv2.VideoCapture(2) # Use index 0 for integrated camera


# Mediapipe hand tracking setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Get screen size
screen_w, screen_h = pyautogui.size()

# Click parameters
click_threshold = 0.05 # Distance to detect click
click_cooldown = 0.8 # Cooldown inbetween clicks
last_click_time = 0
clicking = False

# Smooth cursor movement variables
target_x, target_y = 500, 500
smooth_x, smooth_y = 500, 500
running = True

# Cursor movement thread function
def move_cursor():
    global smooth_x, smooth_y, target_x, target_y, running
    while running:
        # Move cursor smoothly
        smooth_x += (target_x - smooth_x) * 0.25
        smooth_y += (target_y - smooth_y) * 0.25
        try:
            pyautogui.moveTo(smooth_x, smooth_y)
        except Exception:
            pass
        time.sleep(0.015) # Update 60-70 times/ second

# Start cursor movement thread
threading.Thread(target=move_cursor, daemon=True).start()

# Calculate the distance between index and thumb
def distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

# Main loop
while True:
    ret, frame = capture.read() # Capture frames from camera
    if not ret:
        break

    h, w, _ = frame.shape # Frame dimensions
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb) # Process frames for hand landmarks

    # Defining style for landmarks
    landmark_color = mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=3, circle_radius=3)
    # Defining style for connections
    connection_color = mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2)

    # If hand is detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Show landmarks and connections on frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, landmark_color, connection_color)

            # Get index fingertip and thumb tip position
            index_position = hand_landmarks.landmark[8]
            thumb_position = hand_landmarks.landmark[4]

            # Distance between index and thumb
            dist = distance(index_position, thumb_position)

            # Map index position to screen coordinates
            x = int(index_position.x * screen_w)
            y = int(index_position.y * screen_h)

            # Show a circle around index fingertip
            cv2.circle(frame, (int(index_position.x * w), int(index_position.y * h)), 15, (0, 255, 0), 3)

            # Update target for the cursor movement
            target_x, target_y = x, y

            # Detect click gestures
            if dist < click_threshold:
                # Prevent multiple rapid clicks
                if not clicking and (time.time() - last_click_time > click_cooldown):
                    clicking = True
                    pyautogui.click()
                    # Show a text when clicked
                    cv2.putText(frame, "CLICKED!", (0, 50), cv2.FONT_HERSHEY_SIMPLEX,2, (0, 0, 255), 3, cv2.LINE_AA)
                    last_click_time = time.time()
                    print("Clicked!")
            else:
                # Reset when finger are apart
                clicking = False

    # Display video frame
    cv2.imshow("Hand Tracking Cursor Control", frame)
    key = cv2.waitKey(1)
    if key == 27: # ESC key
        break

# Release frame
capture.release()
cv2.destroyAllWindows()
