import cv2
import mediapipe as mp
import numpy as np
import time
import math
from collections import deque

#CAMERA
cap = cv2.VideoCapture(0)

#MEDIAPIPE
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

#CANVAS
canvas = None
preview = None
prev_x = prev_y = 0
start_x = start_y = 0
drawing = False
brush_size = 6
fill_mode = False

#UNDO
undo_stack = deque(maxlen=20)

def save_state():
    undo_stack.append(canvas.copy())

colors = [
    (0,0,0),(255,255,255),(0,0,255),(0,255,0),
    (255,0,0),(0,255,255),(255,0,255),(0,165,255),
    (128,0,128),(203,192,255),(42,42,165),(128,128,128)
]
color_names = [
    "ERASER","WHITE","RED","GREEN","BLUE","YELLOW",
    "MAGENTA","ORANGE","PURPLE","PINK","BROWN","GRAY"
]
current_color = colors[0]

shapes = [
    "FREE","LINE","RECT","CIRCLE","ELLIPSE",
    "TRIANGLE","DIAMOND","PENTAGON",
    "HEXAGON","STAR","ARROW"
]
current_shape = "FREE"

open_colors = False
open_shapes = False

def get_color_name(color):
    for i, c in enumerate(colors):
        if c == color:
            return color_names[i]
    return "CUSTOM"

def fingers_up(hand):
    return [
        hand.landmark[4].x < hand.landmark[3].x,
        hand.landmark[8].y < hand.landmark[6].y,
        hand.landmark[12].y < hand.landmark[10].y,
        hand.landmark[16].y < hand.landmark[14].y,
        hand.landmark[20].y < hand.landmark[18].y
    ]

def draw_top_menu(img):
    w = img.shape[1]
    cv2.rectangle(img,(0,0),(w//2,50),(70,70,70),-1)
    cv2.rectangle(img,(w//2,0),(w,50),(70,70,70),-1)
    cv2.putText(img,"COLORS",(w//4-45,35),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
    cv2.putText(img,"SHAPES",(3*w//4-45,35),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)

def draw_color_menu(img):
    h, w, _ = img.shape
    box_w = w // len(colors)
    for i, col in enumerate(colors):
        x1 = i * box_w
        cv2.rectangle(img,(x1,50),(x1+box_w,100),col,-1)
        cv2.putText(img,color_names[i],(x1+4,85),
                    cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,0,0),1)

def draw_shape_menu(img):
    h, w, _ = img.shape
    box_w = w // len(shapes)
    for i, name in enumerate(shapes):
        x1 = i * box_w
        cv2.rectangle(img,(x1,50),(x1+box_w,100),(90,90,90),-1)
        cv2.putText(img,name,(x1+4,85),
                    cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,255,255),1)

def regular_polygon(cx, cy, r, sides):
    pts = []
    for i in range(sides):
        angle = 2 * math.pi * i / sides - math.pi / 2
        x = int(cx + r * math.cos(angle))
        y = int(cy + r * math.sin(angle))
        pts.append([x, y])
    return np.array(pts, np.int32)


def draw_shape(img, shape, x1, y1, x2, y2):
    thickness = -1 if fill_mode and shape != "LINE" else brush_size
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    r = int(math.hypot(x2 - x1, y2 - y1))

    if shape == "LINE":
        cv2.line(img, (x1, y1), (x2, y2), current_color, brush_size)

    elif shape == "RECT":
        cv2.rectangle(img, (x1, y1), (x2, y2), current_color, thickness)

    elif shape == "CIRCLE":
        cv2.circle(img, (cx, cy), r, current_color, thickness)

    elif shape == "ELLIPSE":
        cv2.ellipse(
            img,
            (cx, cy),
            (abs(x2 - x1), abs(y2 - y1)),
            0, 0, 360,
            current_color,
            thickness
        )

    elif shape == "TRIANGLE":
        pts = regular_polygon(cx, cy, r, 3)
        cv2.fillPoly(img, [pts], current_color) if thickness == -1 \
            else cv2.polylines(img, [pts], True, current_color, brush_size)

    elif shape == "DIAMOND":
        pts = np.array([
            [cx, cy - r],
            [cx + r, cy],
            [cx, cy + r],
            [cx - r, cy]
        ], np.int32)
        cv2.fillPoly(img, [pts], current_color) if thickness == -1 \
            else cv2.polylines(img, [pts], True, current_color, brush_size)

    elif shape == "PENTAGON":
        pts = regular_polygon(cx, cy, r, 5)
        cv2.fillPoly(img, [pts], current_color) if thickness == -1 \
            else cv2.polylines(img, [pts], True, current_color, brush_size)

    elif shape == "HEXAGON":
        pts = regular_polygon(cx, cy, r, 6)
        cv2.fillPoly(img, [pts], current_color) if thickness == -1 \
            else cv2.polylines(img, [pts], True, current_color, brush_size)

    elif shape == "STAR":
        pts = []
        for i in range(10):
            angle = i * math.pi / 5 - math.pi / 2
            radius = r if i % 2 == 0 else r // 2
            x = int(cx + radius * math.cos(angle))
            y = int(cy + radius * math.sin(angle))
            pts.append([x, y])
        pts = np.array(pts, np.int32)
        cv2.fillPoly(img, [pts], current_color) if thickness == -1 \
            else cv2.polylines(img, [pts], True, current_color, brush_size)

    elif shape == "ARROW":
        pts = np.array([
            [x1, y1],
            [x2, y2],
            [x2 - r//4, y2 - r//6],
            [x2 - r//6, y2 - r//4],
        ], np.int32)
        cv2.polylines(img, [pts], False, current_color, brush_size)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame,1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros((h,w,3),dtype=np.uint8)
        preview = canvas.copy()
        save_state()

    draw_top_menu(frame)
    if open_colors:
        draw_color_menu(frame)
    if open_shapes:
        draw_shape_menu(frame)

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame,hand,mp_hands.HAND_CONNECTIONS)

            thumb,index,middle,_,_ = fingers_up(hand)
            x = int(hand.landmark[8].x * w)
            y = int(hand.landmark[8].y * h)

            # ---- MENU ---- #
            if thumb and y < 50:
                open_colors = x < w//2
                open_shapes = x >= w//2

            if index and middle and 50 < y < 100:
                if open_colors:
                    idx = x // (w//len(colors))
                    if idx < len(colors):
                        current_color = colors[idx]
                        open_colors = False
                elif open_shapes:
                    idx = x // (w//len(shapes))
                    if idx < len(shapes):
                        current_shape = shapes[idx]
                        open_shapes = False
            
            #for live patters
            if index and not middle and y > 100:
                if not drawing:
                    start_x, start_y = x, y
                    drawing = True
                    preview = canvas.copy()

                if current_shape == "FREE":
                    cv2.line(canvas,(prev_x or x,prev_y or y),(x,y),
                             current_color,brush_size)
                    prev_x, prev_y = x, y
                else:
                    preview = canvas.copy()
                    draw_shape(preview,current_shape,start_x,start_y,x,y)

            else:
                if drawing:
                    save_state()
                    if current_shape != "FREE":
                        canvas = preview.copy()
                drawing = False
                prev_x = prev_y = 0

    display = preview if drawing and current_shape != "FREE" else canvas

    gray = cv2.cvtColor(display,cv2.COLOR_BGR2GRAY)
    _,inv = cv2.threshold(gray,20,255,cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv,cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame,inv)
    frame = cv2.bitwise_or(frame,display)
    
    #Footer
    tool_mode = "FREE" if current_shape == "FREE" else "SHAPE"
    color_name = get_color_name(current_color)
    fill_text = "ON" if fill_mode else "OFF"

    overlay = frame.copy()
    cv2.rectangle(overlay,(0,h-22),(w,h),(0,0,0),-1)
    frame = cv2.addWeighted(overlay,0.35,frame,0.65,0)

    cv2.putText(
        frame,
        f"Tool:{tool_mode} | Shape:{current_shape} | Color:{color_name} | "
        f"Fill:{fill_text} | Brush:{brush_size} | "
        f"U:Undo | F:Fill | C:Clear | S:Save | Q:Quit",
        (8,h-7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (220,220,220),
        1
    )

    cv2.imshow("Air Canvas - Fill Shapes Added",frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('+'), ord('=')):
        brush_size = min(brush_size + 1, 30)
    if key == ord('-'):
        brush_size = max(brush_size - 1, 1)
    if key == ord('f'):
        fill_mode = not fill_mode
    if key == ord('u') and len(undo_stack) > 1:
        undo_stack.pop()
        canvas = undo_stack[-1].copy()
        preview = canvas.copy()
    if key == ord('c'):
        save_state()
        canvas = np.zeros((h,w,3),dtype=np.uint8)
        preview = canvas.copy()
    if key == ord('s'):
        cv2.imwrite(f"air_canvas_{int(time.time())}.png",canvas)
        print("Saved")
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
