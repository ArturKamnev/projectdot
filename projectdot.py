import cv2
import mediapipe as mp
import pyautogui
import math
import time


HOLD_THRESHOLD      = 0.5   
CLICK_COOLDOWN      = 0.3   
RIGHT_CLICK_COOLDOWN= 0.3   
MOVE_SCALE          = 1.2 

def main():
    pyautogui.FAILSAFE = False
    screen_w, screen_h = pyautogui.size()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    
    prev_x = prev_y = None
    alpha = 0.2

    
    pinch_active       = False
    pinch_start        = 0.0
    is_holding         = False
    last_click_time    = 0.0

    
    pinch_active_mid   = False
    last_right_click   = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            
            small = cv2.resize(frame, (320, 240))
            small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            res = hands.process(small_rgb)
            now = time.time()

            if res.multi_hand_landmarks:
                for lm, hm in zip(res.multi_hand_landmarks, res.multi_handedness):
                    label = hm.classification[0].label 

                    
                    for tip in (4, 8, 12, 16, 20):
                        x = int(lm.landmark[tip].x * w)
                        y = int(lm.landmark[tip].y * h)
                        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

                    
                    x0, y0 = lm.landmark[0].x, lm.landmark[0].y
                    x9, y9 = lm.landmark[9].x, lm.landmark[9].y
                    hand_size = math.hypot(x0 - x9, y0 - y9)
                    threshold = 0.3 * hand_size

                    if label == 'Right':
                        
                        ix = lm.landmark[8].x
                        iy = lm.landmark[8].y

                        sx_norm = (ix - 0.5) * MOVE_SCALE + 0.5
                        sy_norm = (iy - 0.5) * MOVE_SCALE + 0.5
                        sx_norm = max(0.0, min(1.0, sx_norm))
                        sy_norm = max(0.0, min(1.0, sy_norm))

                        ix_scr = sx_norm * screen_w
                        iy_scr = sy_norm * screen_h

                        if prev_x is None:
                            sx, sy = ix_scr, iy_scr
                        else:
                            sx = alpha * ix_scr + (1 - alpha) * prev_x
                            sy = alpha * iy_scr + (1 - alpha) * prev_y

                        pyautogui.moveTo(sx, sy, _pause=False)
                        prev_x, prev_y = sx, sy

                    else:
                        
                        x4, y4 = lm.landmark[4].x, lm.landmark[4].y
                        x12, y12 = lm.landmark[12].x, lm.landmark[12].y
                        dist_mid = math.hypot(x4 - x12, y4 - y12)

                        if dist_mid < threshold:
                            if not pinch_active_mid:
                                pinch_active_mid = True
                        else:
                            if pinch_active_mid:
                                if now - last_right_click >= RIGHT_CLICK_COOLDOWN:
                                    pyautogui.click(button='right')
                                    last_right_click = now
                                pinch_active_mid = False

                        
                        x8, y8 = lm.landmark[8].x, lm.landmark[8].y
                        dist_idx = math.hypot(x4 - x8, y4 - y8)

                        if dist_idx < threshold:
                            if not pinch_active:
                                pinch_active = True
                                pinch_start  = now
                        else:
                            if pinch_active:
                                duration = now - pinch_start
                                
                                if duration >= HOLD_THRESHOLD:
                                    if is_holding:
                                        pyautogui.mouseUp()
                                else:
                                    
                                    if now - last_click_time >= CLICK_COOLDOWN:
                                        pyautogui.click()
                                        last_click_time = now
                                pinch_active = False
                                is_holding  = False

                        
                        if pinch_active and not is_holding and (now - pinch_start) >= HOLD_THRESHOLD:
                            pyautogui.mouseDown()
                            is_holding = True

            cv2.imshow('Finger Control (v4)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
