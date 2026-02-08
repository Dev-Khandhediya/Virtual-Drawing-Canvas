import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
canvas = None
prev_x, prev_y = None, None

is_tracking = False
last_toggle_time = 0
cooldown = 0.8
current_color = (0, 0, 255)

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=50, detectShadows=False
)

clear_hover = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    toolbar_h = 100

    
    bx1, by1, bx2, by2 = 10, 20, 110, 70     # CLEAR button
    tx1, ty1, tx2, ty2 = 120, 20, 230, 70    # PEN button

    colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (0,0,0)]
    color_names = ["RED", "GREEN", "BLUE", "YELLOW", "ERASER"]

    palette_start_x = 280
    palette_box_w = 70
    palette_gap = 10

    total_palette_w = len(colors)*palette_box_w + (len(colors)-1)*palette_gap
    if palette_start_x + total_palette_w > w - 10:
        palette_start_x = w - total_palette_w - 10

    sx1, sy1 = 20, toolbar_h + 10
    sx2, sy2 = w - 20, h - 20

    clear_hover = False

    fgmask = fgbg.apply(frame)
    kernel = np.ones((5,5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.dilate(fgmask, kernel, iterations=2)

    contours, _ = cv2.findContours(
        fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 1500:
            hull = cv2.convexHull(cnt)
            cx, cy = min(hull, key=lambda p: p[0][1])[0]

            cv2.circle(frame, (cx, cy), 12, (255,255,255), 2)
            cv2.circle(frame, (cx, cy), 2, (255,255,255), -1)

            curr_time = time.time()

            if bx1 < cx < bx2 and by1 < cy < by2:
                canvas[:] = 0
                prev_x = prev_y = None
                clear_hover = True

            elif tx1 < cx < tx2 and ty1 < cy < ty2:
                if curr_time - last_toggle_time > cooldown:
                    is_tracking = not is_tracking
                    last_toggle_time = curr_time
                    prev_x = prev_y = None

            elif by1 < cy < by2:
                prev_x = prev_y = None
                for i, col in enumerate(colors):
                    px1 = palette_start_x + i * (palette_box_w + palette_gap)
                    px2 = px1 + palette_box_w
                    if px1 < cx < px2:
                        current_color = col

            elif is_tracking and sx1 < cx < sx2 and sy1 < cy < sy2:
                if prev_x is not None and np.hypot(cx-prev_x, cy-prev_y) < 70:
                    cv2.line(canvas, (prev_x, prev_y),
                             (cx, cy), current_color, 10)
                prev_x, prev_y = cx, cy
            else:
                prev_x = prev_y = None

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
    bg = cv2.bitwise_and(frame, frame, mask=mask)
    output = cv2.add(bg, canvas)

    cv2.rectangle(output, (0, 0), (w, toolbar_h), (40,40,40), -1)

    clear_bg = (0,0,0) if clear_hover else (220,220,220)
    clear_txt = (255,255,255) if clear_hover else (0,0,0)

    cv2.rectangle(output, (bx1,by1), (bx2,by2), clear_bg, -1)
    cv2.putText(output, "CLEAR", (bx1+20,by1+35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, clear_txt, 2)

    pen_col = (0,255,0) if is_tracking else (120,120,120)
    cv2.rectangle(output, (tx1,ty1), (tx2,ty2), pen_col, -1)
    cv2.putText(output,
                "ON" if is_tracking else "OFF",
                (tx1+18,ty1+35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    for i, col in enumerate(colors):
        px1 = palette_start_x + i * (palette_box_w + palette_gap)
        px2 = px1 + palette_box_w

        if current_color == col:
            cv2.rectangle(output, (px1-4,by1-4),
                          (px2+4,by2+4), (255,255,255), 2)

        cv2.rectangle(output, (px1,by1), (px2,by2), col, -1)
        cv2.rectangle(output, (px1,by1), (px2,by2), (200,200,200), 1)

        txt_col = (255,255,255) if col==(0,0,0) else (0,0,0)
        cv2.putText(output, color_names[i],
                    (px1+5, by2-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, txt_col, 1)

    cv2.rectangle(output, (sx1,sy1), (sx2,sy2), (255,255,255), 2)
    cv2.putText(output, "DEV CANVAS",
                (sx1+15, sy1+35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("DEV STUDIO ", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
