# echo_server.py
#-*- coding:utf-8 -*-
import os

import socket
import json
import time

import mediapipe as mp
import cv2
import numpy as np
import matplotlib.pyplot as plt
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

classes = ["person", "bicycle", "car", "motorcycle",
            "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
            "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
            "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
            "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
            "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
            "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
            "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
            "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ]


len(classes)

#상황별 필요 객체 목록
meal_con = ["bottle", "bowl", "cup"]
media_con = ["remote", "cell phone"]
work_con = ["keyboard", "laptop", "book"]

#결과 송출 부분
meal_check = 0
score_con = 0
# Yolo load
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


#calculate angles
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
                
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
                
    if angle > 180.0:
        angle = 360-angle
                    
    return angle
        
#calculate Y diff
def calculate_Y_diff_abs(a, b):
    a = np.array(a)[1]
    b = np.array(b)[1]
    return abs(a-b)

def calculate_Y_diff(a, b):
    a = np.array(a)[1]
    b = np.array(b)[1]
    return (a-b)




# 통신 정보 설정
IP = '192.168.0.14'
PORT = 8080
SIZE = 1024
ADDR = (IP, PORT)




#비디오 설정
stage = {
    "CurrentMoment" : "test currnet",
    "PreviousMoment": "test upcoming"
}
temp = cv2.VideoCapture(cv2.CAP_DSHOW) #카메라
stage['CurrentMoment'] = "Initial"
result = json.dumps(stage)
motion = "-"

state_cnt = 0

NOW = []

present = []
past = []
result = []

meal_cnt = 0
media_cnt = 0
work_cnt = 0



# 서버 소켓 설정
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind(ADDR)  # 주소 바인딩
    server_socket.listen()  # 클라이언트의 요청을 받을 준비


    # 루프 진입
    while temp.isOpened():    
        
        #setup mp
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:          
        
            
            #사진 촬영
            temp_ret, temp_frame = temp.read()
            #사진 저장
            cv2.imwrite("temp.png", temp_frame)
            # time.sleep(0.5)
            #저장된 이미지
            img = cv2.imread("temp.png") 
            ret, frame = temp.read()
                
            #recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            #make detection
            results = pose.process(image)
            #recoloring back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            #extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                            
                #calculate (shoulder-hip-knee)
                right_angle = calculate_angle(right_shoulder, right_hip, right_knee)
                left_angle = calculate_angle(left_shoulder, left_hip, left_knee)
                #calculate (hip-knee-ankle)
                right_angle_leg = calculate_angle(right_hip, right_knee, right_ankle)
                left_angle_leg = calculate_angle(left_hip, left_knee, left_ankle)
                
                
                if calculate_Y_diff_abs(right_shoulder, right_hip)<0.05 or calculate_Y_diff_abs(left_shoulder, left_hip)<0.05:
                    motion = "lie"
                else:
                        
                    if (right_angle>140 and (left_angle>70 and left_angle<=140)) or (left_angle>140 and (right_angle>70 and right_angle<=140)): 
                        motion = "sit"
                        
                    elif (right_angle>140 and left_angle<75) or (left_angle>140 and right_angle<75): 
                        motion = "sit"
                        
                    elif ((right_angle>70 and right_angle<=140)and(left_angle<=75)) or ((left_angle>70 and left_angle<=140)and(right_angle<=75)): 
                        motion = "sit"
                                    
                    elif (right_angle>70 and right_angle<=140) and (left_angle>70 and left_angle<=140): 
                        if calculate_Y_diff(right_knee, right_hip)<=0.1:
                            motion = "sit"
                        else:
                            motion = "stand"
                            
                    elif right_angle>140 and left_angle>140:
                        motion = "stand"
                        
                    elif right_angle<=100 and left_angle<=100: 
                        if calculate_Y_diff(right_knee, right_hip)<=0.1 or calculate_Y_diff(left_knee, left_hip)<=0.1:
                            motion = "sit"              
            except:
                pass


        
        
        
        
            # 동작1 sit
            if motion == "sit": #마지막에 sit으로 바꾸자
                        #img load
                img = cv2.imread("temp.png")
                #img = cv2.resize(img, None, fx=0.4, fy=0.4)
                height, width, channels = img.shape

                blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layers)

                class_ids = []
                confidences = []
                boxes = []
                
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]

                        if confidence > 0.5:
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)

                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                class_in = []
                box = []
                

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.4)

                font = cv2.FONT_HERSHEY_PLAIN
                
                meal_cnt = 0
                media_cnt = 0
                work_cnt = 0
                present = []
                
                for i in range(len(boxes)):
                    
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(classes[class_ids[i]])
                        
                        present.append(label)
                        
                        print(f"class_ids: {label} x : {x} y : {y}")
                        color = colors[i]
                        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(img, label, (x, y+30), font, 2, color, 2)
        
                #--------------------------------------------------------------------------        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                #변화 없으면    
                if present == past:
                    
                    #3턴째 변화 없으면
                    if state_cnt >= 3:
                        print("motion : " + motion)
                        print("CurrentMoment : " + stage["CurrentMoment"])
                        print("NO DIFFERENCE")
                        print("============================")
            
                        #다시 처음부터
                        continue
                    
                    #아직 3턴 아니면
                    state_cnt += 1
                    #그냥 현재걸로 계산
                    #result = present
                        
                        
                        
                        
                        
                        
                    
                #변화 있으면    
                else:    
                    state_cnt = 0
                    
                    #새로 생긴걸로 계산
                    result = list(set(present) - set(past))
                    
                    if len(past) > len(present):
                        result = present
                    
                    
                    
                    
                    
                    
                
                # time.sleep(0.2)
                
                
                
                
                
                
                
                
                
                
                print("----------")
                print("state_cnt : ")
                print(state_cnt)
                print("present : ")
                print(present)
                print("past : ")
                print(past)
                print("result : ")
                print(result)
                print("----------")
                
                for i in range(len(result)):
                    
                    if result[i] in meal_con:
                        meal_cnt += 1
                        
                    elif result[i] in media_con:
                        media_cnt += 1
                        
                    elif result[i] in work_con:
                        work_cnt += 1
                        
                    else:
                        pass
                
                
                
                print("meal : " + str(meal_cnt))
                print("media : " + str(media_cnt))
                print("work : " +str(work_cnt))
                
                
                
                
                
                
                
                
                
                max_cnt = ""
                
                if meal_cnt > media_cnt:
                    if meal_cnt > work_cnt:
                        max_cnt = "meal"
                    elif meal_cnt == work_cnt:
                        max_cnt = "initial"
                    elif meal_cnt < work_cnt:
                        max_cnt = "work"
                        
                elif meal_cnt == media_cnt:
                    if media_cnt > work_cnt:
                        max_cnt = "initial"
                    elif media_cnt == work_cnt:
                        max_cnt = "initial"
                    elif media_cnt < work_cnt:
                        max_cnt = "work"
                        
                elif meal_cnt < media_cnt:
                    if media_cnt > work_cnt:    
                        max_cnt = "media" 
                    elif media_cnt == work_cnt:
                        max_cnt = "initial"
                    elif media_cnt < work_cnt:
                        max_cnt = "work"
                


                
                print("max : " + max_cnt)
        
        
                #-------------------------------------------------------------------------- 
        

                if max_cnt == "meal":
                    NOW.insert(0, "meal")
                                
                elif max_cnt == "work":
                        NOW.insert(0, "work")
                    
                elif max_cnt == "media":
                    NOW.insert(0, "media")
                    
                else:
                    pass
                        










            #동작2 lie -> sleep
            elif motion == "lie":
                NOW.insert(0, "sleep")


            #동작3 stand -> 전 상태 유지
            else:
                print("wait => " + stage["CurrentMoment"])

                

    
        
        
        
            # time.sleep(0.2)


        
                        
            if(len(NOW) > 3):
                print("pop")
                NOW.pop()       
            print("NOW : ")
            print(NOW)

            
            if(NOW.count("work")  == 2):
                stage["CurrentMoment"] = "work"
                
            elif(NOW.count("media")  == 2):
                stage["CurrentMoment"] = "media"
                
            elif(NOW.count("meal")  == 2):
                stage["CurrentMoment"] = "meal"
                
            elif(NOW.count("sleep")  == 2):
                stage["CurrentMoment"] = "sleep"
                
            else:
                print("-------------------------------------------------")
                continue
                
                
        
            client_socket, client_addr = server_socket.accept()
            # 수신대기, 접속한 클라이언트 정보 (소켓, 주소) 반환

            msg = client_socket.recv(SIZE)
            # 클라이언트가 보낸 메시지 반환

            print("[{}] message : {}".format(client_addr,msg))
            # 클라이언트가 보낸 메시지 출력
            
            
            
            result = json.dumps(stage)
            client_socket.sendall(result.encode())  # 클라이언트에게 응답
            print("전송 성공")
        
                
            print("motion : " + motion)
            print("CurrentMoment : " + stage["CurrentMoment"])

            
            meal_check = 0
            score_con = 0
            stage["CurrentMoment"] = "initial"        
                    
            print("-------------------------------------------------")
                    
            past = present
            
            #사진 삭제
            os.remove("temp.png")
            print("PNG REMOVED")

            time.sleep(0.2)
            
            

            
            
        
    #client_socket.close()  # 클라이언트 소켓 종료
        
    