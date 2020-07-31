from models import *
from utils import *
from sort import *
from PIL import Image
import cv2
import os, sys
import heapq
import math
import statistics as stats
from collections import Counter

import warnings
warnings.filterwarnings("ignore")

try:
    if(sys.argv[1] != '0' and sys.argv[1] != '1'):
        if os.path.exists(sys.argv[1]):
            input_rsc = sys.argv[1]
        else:
            print('Wrong VIDEO_PATH, file does not exist')
            raise SystemExit(f"Usage: python {sys.argv[0]} CAM_INDEX or python {sys.argv[0]} VIDEO_PATH")
    else:
        input_rsc = int(sys.argv[1])

except Exception as e:
    print('No input source')
    raise SystemExit(f"Usage: python {sys.argv[0]} CAM_INDEX or python {sys.argv[0]} VIDEO_PATH")

gpu = False
try:
    if sys.argv[2] == '-gpu':
        gpu = True
    else:
        print("Wrong flag: " + sys.argv[2])
        print("(RECOMMENDED) To use GPU, just add -gpu flag at the end")
        print("Running with CPU")
except:
    print("Running with CPU")
    print("(RECOMMENDED) To use GPU, just add -gpu flag at the end")
    pass

custom_lenet_class_path = "config/alphabet.names"
custom_lenet_weights_path = "weights/custom_lenet_100_02.weights.weights"

# load weights and set defaults
yolo_config_path='config/yolov3_face_hand.cfg'
yolo_weights_path='weights/yolov3_face_hand.weights'
yolo_class_path='config/face_hand.names'

img_size = 608
conf_thresh = 0.2
nms_thresh = 0.4

custom_lenet_thresh = 0.75

yolo_classes = load_classes(yolo_class_path)
custom_lenet_classes = load_classes(custom_lenet_class_path)

# load model and put into eval mode
darknet_model = Darknet(yolo_config_path, img_size=img_size)
darknet_model.load_weights(yolo_weights_path)

custom_lenet_model = CustomModel(n_output=len(custom_lenet_classes))
custom_lenet_model.load_weights(custom_lenet_weights_path, gpu)

if gpu:
    try:
        darknet_model.cuda()
    except:
        darknet_model.cpu()
    try:
        custom_lenet_model = custom_lenet_model.cuda()
        print("Running with GPU")
    except Exception as e:
        gpu = False
        custom_lenet_model.cpu()

darknet_model.eval()
custom_lenet_model.eval()

colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

cap = cv2.VideoCapture(input_rsc)
fps = cap.get(cv2.CAP_PROP_FPS)

if cap.isOpened():
    ret, frame = cap.read()
else:
    ret = False

mot_tracker = Sort() 

cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stream', (600,400))
cv2.moveWindow('Stream', 0, 0)

cv2.namedWindow('Word',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Word', (2000,300))
cv2.moveWindow('Word', 0, 1000)

text_position = (50,180)
black = np.zeros((300,2000,3),np.uint8)

frames = 0
word = ''
last_center = (0,0)

last_detected_letter = ''
no_hand_detection_counter = 0
letter_detection_array = []
if gpu:
    clear_array_count = 20
    letter_detection_wait = 20
    no_detection_time = 40
    letter_array_wait = 12
    min_array_quantity = 6
    
if not gpu:
    clear_array_count = 5
    letter_detection_wait = 3
    no_detection_time = 6
    letter_array_wait = 6
    min_array_quantity = 2

while ret:
    ret, frame = cap.read()
    frames += 1
    flag = 0
    
    try:
        frame_shape = frame.shape
    except:
        if len(word) >= 3:
            save_full_word_with_gtts(word)
            say_full_word_with_pyttsx3(word)
            print('Finished video')
            exit(0)

    if frames % 1 == 0:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        pilimg = Image.fromarray(frame)
        yolo_detections = yolo_detector(pilimg, img_size, darknet_model, conf_thresh, nms_thresh, gpu)
        img = np.array(pilimg)
        pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x
        if yolo_detections is not None:
            tracked_objects = mot_tracker.update(yolo_detections.cpu().double())

            a = tracked_objects.tolist()
            a = sorted(a, key=lambda a_entry: a_entry[6], reverse=True) 
            a = sorted(a, key=lambda a_entry: a_entry[5], reverse=True)
            try:
                tracked_hand = a[0]
                x1, y1, x2, y2, obj_id, cls_pred, precision = tracked_hand
                if cls_pred == 1.0:
                    precision = str(round(precision * 100, 2)) + "%"
                    
                    box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
                    box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
                    y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
                    x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
                    color = colors[int(obj_id) % len(colors)]

                    width_tresh = box_w / frame_shape[0] * 100
                    height_thresh = box_h / frame_shape[1] * 100

                    cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 2)

                    current_center = (x1 + box_w - int(box_w/2), y1 + box_h - int(box_h/2))
                    a = (current_center[0]-last_center[0])**2
                    b = (current_center[1]-last_center[1])**2
                    d = math.sqrt(abs(a-b))

                    last_center = current_center
                    try:
                        no_hand_detection_counter = 0

                        hand = frame[y1 : y1 + box_h, x1: x1 + box_w]

                        max_classes = classify_letter(hand, custom_lenet_model, custom_lenet_classes, gpu)
                        if (max_classes[0][1] > custom_lenet_thresh) and (d < 1.5):

                            accuracy = round(max_classes[0][1] * 100, 2)
                            accuracies = (round(max_classes[0][1] * 100, 2), round(max_classes[1][1] * 100, 2), round(max_classes[2][1] * 100, 2))
                            detect_letters = (str(max_classes[0][0]), str(max_classes[1][0]), str(max_classes[2][0]))

                            if (last_detected_letter == detect_letters[0]): 
                                detected_letter_counter += 1
                            if (last_detected_letter != detect_letters[0]): 
                                detected_letter_counter = 0
                            if (last_detected_letter == ''): 
                                last_detected_letter = detect_letters[0] #first letter when no detected last letter
                            
                            print('Letter detected: ', detect_letters[0], ' - Accuracy: ', str(accuracies[0]) + '%')

                            letter_detection_array.append(detect_letters[0])
                            
                            #WHEN WORD IS EMPTY
                            if(word == ''):
                                black = updateText(word)

                                #ADD MOST COMMON IN ARRAY
                                if len(letter_detection_array) > letter_array_wait:
                                    ctr = Counter(np.array(letter_detection_array).ravel())
                                    letter_detection_array = []
                                    letters_to_add = ctr.most_common(2)
                                    if letters_to_add[0] != [] and letters_to_add[0][1] > min_array_quantity:
                                        word = word + letters_to_add[0][0]
                                        flag = 1
                                        last_detected_letter = letters_to_add[0][0]
                                        detected_letter_counter = 0
                                        letter_detection_array = []
                                        print('WORD: ', word)
                                        black = updateText(word)

                                    #ADD SECOND MOST COMMON IN ARRAY
                                    elif letters_to_add[1] != [] and letters_to_add[1][1] > min_array_quantity:
                                        word = word + letters_to_add[1][0]
                                        flag = 1
                                        last_detected_letter = letters_to_add[1][0]
                                        detected_letter_counter = 0
                                        letter_detection_array = []
                                        print('WORD: ', word)
                                        black = updateText(word)

                                #ADD BY LETTER COUNTER
                                if detected_letter_counter > letter_detection_wait:
                                    word = word + detect_letters[0]
                                    flag = 1
                                    detected_letter_counter = 0
                                    letter_detection_array = []
                                    print('WORD: ', word)
                                    black = updateText(word)

                            #WHEN WORD IS NOT EMPTY
                            if (word != ''):
                                if len(letter_detection_array) > letter_array_wait: #and last_detected_letter !=detect_letters[0]:
                                    ctr = Counter(np.array(letter_detection_array).ravel())
                                    letter_detection_array = []
                                    letters_to_add = ctr.most_common(2)
                                    #ADD MOST COMMON IN ARRAY
                                    if word[-1] != letters_to_add[0][0] and letters_to_add[0] != [] and letters_to_add[0][1] > min_array_quantity:
                                        word = word + letters_to_add[0][0]
                                        flag = 1
                                        last_detected_letter = letters_to_add[0][0]
                                        detected_letter_counter = 0
                                        letter_detection_array = []
                                        print('WORD: ', word)
                                        black = updateText(word)
                                    
                                    #ADD SECOND MOST COMMON IN ARRAY
                                    elif word[-1] != letters_to_add[1][0] and letters_to_add[1] != [] and letters_to_add[1][1] > min_array_quantity:
                                        word = word + letters_to_add[1][0]
                                        flag = 1
                                        last_detected_letter = letters_to_add[1][0]
                                        detected_letter_counter = 0
                                        letter_detection_array = []
                                        print('WORD: ', word)
                                        black = updateText(word)

                                #ADD BY LETTER COUNTER
                                if detected_letter_counter > letter_detection_wait and word[-1] != detect_letters[0]:
                                    word = word + detect_letters[0]
                                    flag = 1
                                    detected_letter_counter = 0
                                    letter_detection_array = []
                                    print('WORD: ', word)
                                    black = updateText(word)

                            #NO HAND DETECTION, CLEAN ARRAY
                            if (no_hand_detection_counter > clear_array_count):
                                if(len(letter_detection_array) > 15):
                                    letter_to_add = stats.mode(letter_detection_array)
                                    word = word + letter_to_add
                                    flag = 1
                                    last_detected_letter = letter_to_add
                                    detected_letter_counter = 0
                                    print('WORD: ', word)
                                    black = updateText(word)
                                letter_detection_array = []

                            last_detected_letter = detect_letters[0]
                        else:
                            pass
                    except:
                        pass
            except:
                pass
        else:
            no_hand_detection_counter += 1
            
            #NO HAND DETECTION, CLEAN ARRAY
            if (no_hand_detection_counter > clear_array_count):
                letter_detection_array = []

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('Word', black)
        cv2.imshow('Stream', frame)
        ch = 0xFF & cv2.waitKey(1)
        if ch == ord('q'):
            break

        #Delete last letter (Backspace)
        if ch == 8: 
            try:
                word = word[:-1]
                black = updateText(word)
                detected_letter_counter = 0
                letter_detection_array = []
            except:
                pass

        #Delete word ('B' key)
        if ch == ord('b'):
            word = ''
            black = updateText(word)
            detected_letter_counter = 0
            letter_detection_array = []
            flag = 0

        #Put a space after word (Space)
        if ch == ord(' '):
            word = word + ' '
            black = updateText(word)
            detected_letter_counter = 0
            letter_detection_array = []

        #Say full word (Enter)
        if ch == 13:
            try:
                if len(word) >= 3:
                    save_full_word_with_gtts(word)
                    say_full_word_with_pyttsx3(word)
                    flag = 2
                    word = ''
                    black = updateText(word)
            except:
                pass

        #Entering dynamic letters on keyboard
        if ch == ord('g'):
            word = word + 'G'
            last_detected_letter = 'G'
            detected_letter_counter = 0
            letter_detection_array = []
            print('WORD: ', word)
            flag = 1
            black = updateText(word)

        if ch == ord('j'):
            word = word + 'J'
            last_detected_letter = 'J'
            detected_letter_counter = 0
            letter_detection_array = []
            print('WORD: ', word)
            flag = 1
            black = updateText(word)

        if ch == 209:
            word = word + 'Ñ'
            last_detected_letter = 'Ñ'
            detected_letter_counter = 0
            letter_detection_array = []
            print('WORD: ', word)
            flag = 1
            black = updateText(word)

        if ch == ord('s'):
            word = word + 'S'
            last_detected_letter = 'S'
            detected_letter_counter = 0
            letter_detection_array = []
            print('WORD: ', word)
            flag = 1
            black = updateText(word)

        if ch == ord('x'):
            word = word + 'X'
            last_detected_letter = 'X'
            detected_letter_counter = 0
            letter_detection_array = []
            print('WORD: ', word)
            flag = 1
            black = updateText(word)

        if ch == ord('z'):
            word = word + 'Z'
            last_detected_letter = 'Z'
            detected_letter_counter = 0
            letter_detection_array = []
            print('WORD: ', word)
            flag = 1
            black = updateText(word)
        
        if len(word) >= 3 and no_hand_detection_counter > no_detection_time :
            save_full_word_with_gtts(word)
            say_full_word_with_pyttsx3(word)
            flag = 2
            word = ''
            black = updateText(word)

cv2.destroyAllWindows()