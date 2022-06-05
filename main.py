from re import X
import cv2
import time
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import math
import numpy as np


BODY_PARTS_BODY_25 = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                      5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
                      10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
                      15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
                      20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel", 25: "Background"}

POSE_PAIRS_BODY_25 = [[0, 1], [0, 15], [0, 16], [1, 2], [1, 5], [1, 8], [8, 9], [8, 12], [9, 10], [12, 13], [2, 3],
                      [3, 4], [5, 6], [6, 7], [10, 11], [13, 14], [15, 17], [16, 18], [14, 21], [19, 21], [20, 21],
                      [11, 24], [22, 24], [23, 24]]


BODY_PARTS_MPI = {0: "Head", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                  5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                  10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "Chest",
                  15: "Background"}

POSE_PAIRS_MPI = [[0, 1], [1, 2], [1, 5], [1, 14], [2, 3], [3, 4], [5, 6],
                  [6, 7], [8, 9], [9, 10], [11, 12], [12, 13], [14, 8], [14, 11]]

protoFile_body_25="./model/pose_deploy.prototxt"
weightsFile_body_25="./model/pose_iter_584000.caffemodel"

protoFile_mpi_faster="./model/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile_mpi="./model/pose_iter_160000.caffemodel"

# 키포인트를 저장할 빈 리스트
points = []
point_in=[]

# 동작 예측 결과 (arabesque, passe, plie)
pose=""

# 동작에 대한 coaching text
msg=" "

def angle_between(p1, p2):  #두점 사이의 각도:(getAngle3P 계산용) 시계 방향으로 계산한다. P1-(0,0)-P2의 각도를 시계방향으로
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    res = np.rad2deg((ang1 - ang2) % (2 * np.pi))
    return res

def getAngle3P(p1, p2, p3, direction="CW"): #세점 사이의 각도 1->2->3
    pt1 = (p1[0] - p2[0], p1[1] - p2[1])
    pt2 = (p3[0] - p2[0], p3[1] - p2[1])
    res = angle_between(pt1, pt2)
    res = (res + 360) % 360
    if direction == "CCW":    #반시계방향
        res = (360 - res) % 360
    return res

def feedback_func(): #coaching text 생성
    global point_in
    global pose
    global msg
    
    if pose=='passe':
        ld=getAngle3P((point_in[16], point_in[17]),(point_in[18], point_in[19]),(point_in[20], point_in[21]),"CCW")
        rd=getAngle3P((point_in[22], point_in[23]),(point_in[24], point_in[25]),(point_in[26], point_in[27]))
        if ld<rd and ld > 45:
            msg="Raise left leg " +str(round(ld-45,1))+"deg more"
        elif rd< ld and rd> 45:
            msg="Raise right leg " +str(round(rd-45,1))+"deg more"
        else:
            msg="Good"
            
        print('ld',ld, 'rd',rd)
        
    if pose=='plie':
        ld=getAngle3P((point_in[16], point_in[17]),(point_in[18], point_in[19]),(point_in[20], point_in[21]))
        rd=getAngle3P((point_in[22], point_in[23]),(point_in[24], point_in[25]),(point_in[26], point_in[27]),"CCW")
        if ld > 100 or rd > 100:
            msg="Bend your knee more " +str(round(max(ld,rd)-100,1))+"deg more"
        else:
            msg="Good"
            
        print('ld',ld, 'rd',rd)
        
    if pose=='arabesque':
        leg1=getAngle3P((point_in[18], point_in[19]),((point_in[22]+point_in[16])/2, (point_in[23]+point_in[17])/2),(point_in[24], point_in[25]))
        leg2=getAngle3P((point_in[18], point_in[19]),((point_in[22]+point_in[16])/2, (point_in[23]+point_in[17])/2),(point_in[24], point_in[25]),"CCW")
        if leg1>leg2 and leg2<90:
            msg="Raise left leg " +str(round(90-leg2,1))+"deg more"
        if leg2>leg1 and leg1<90:
            msg="Raise right leg " +str(round(90-leg1,1))+"deg more"
        else:
            msg="Good"
            
        print('leg1',leg1, 'leg2', leg2)
    
def predict_pose(model_name,frame_width, frame_height): # xgboost classifier로 동작 분류
    global points
    global pose
    global point_in
    x=[]
    if model_name == 'BODY_25':
        temp=points[:8]+points[9:15]
        for i in temp:
            if i==None:
                x.append(0)
                x.append(0)
            else:
                x.append(i[0]/frame_width)
                x.append(i[1]/frame_height)
        if temp[-2] != None and temp[-1]!=None:
            chest=(temp[-2][0]+temp[-1][0])/2
            x.append(chest/frame_width)
            chest=(temp[-2][1]+temp[-1][1])/2
            x.append(chest/frame_height)
        else:
            x.append(0)
            x.append(0)
            
    if model_name == 'MPI':
        temp=points[:15]
        for i in temp:
            if i==None:
                x.append(0)
                x.append(0)
            else:
                x.append(i[0]/frame_width)
                x.append(i[1]/frame_height)
    point_in=x
    clf=xgb.XGBClassifier()
    booster=xgb.Booster()
    booster.load_model('xgbmodel.json')
    clf._Booster=booster
    clf._le=LabelEncoder().fit(['label0','label1','label2'])
    y_pred=clf.predict(x)

    if y_pred=='label0':
        pose='arabesque'
    elif y_pred=='label1':
        pose='passe'
    elif y_pred=='label2':
        pose='plie'

    print(y_pred)


def output_keypoints(frame, proto_file, weights_file, threshold, model_name, BODY_PARTS): # open pose 사용하여 keypoints 예측
    global points

    # 네트워크 불러오기
    net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

    # 입력 이미지의 사이즈 정의
    image_height = 368
    image_width = 368

    # 네트워크에 넣기 위한 전처리
    input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (image_width, image_height), (0, 0, 0), swapRB=False, crop=False)

    # 전처리된 blob 네트워크에 입력
    net.setInput(input_blob)

    # 결과 받아오기
    out = net.forward()
    # The output is a 4D matrix :
    # The first dimension being the image ID ( in case you pass more than one image to the network ).
    # The second dimension indicates the index of a keypoint.
    # The model produces Confidence Maps and Part Affinity maps which are all concatenated.
    # For COCO model it consists of 57 parts – 18 keypoint confidence Maps + 1 background + 19*2 Part Affinity Maps. Similarly, for MPI, it produces 44 points.
    # We will be using only the first few points which correspond to Keypoints.
    # The third dimension is the height of the output map.
    out_height = out.shape[2]
    # The fourth dimension is the width of the output map.
    out_width = out.shape[3]

    # 원본 이미지의 높이, 너비를 받아오기
    frame_height, frame_width = frame.shape[:2]

    # 포인트 리스트 초기화
    points = []

    print(f"\n============================== {model_name} Model ==============================")
    for i in range(len(BODY_PARTS)):

        # 신체 부위의 confidence map
        prob_map = out[0, i, :, :]

        # 최소값, 최대값, 최소값 위치, 최대값 위치
        min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

        # 원본 이미지에 맞게 포인트 위치 조정
        x = (frame_width * point[0]) / out_width
        x = int(x)
        y = (frame_height * point[1]) / out_height
        y = int(y)

        if prob > threshold:  # [pointed]
            cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, lineType=cv2.LINE_AA)

            points.append((x, y))
            print(f"[pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")

        else:  # [not pointed]
            cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, lineType=cv2.LINE_AA)

            points.append(None)
            print(f"[not pointed] {BODY_PARTS[i]} ({i}) => prob: {prob:.5f} / x: {x} / y: {y}")

    predict_pose(model_name,frame_width, frame_height)
    feedback_func()
    return frame

def output_keypoints_with_lines(frame, POSE_PAIRS):
    global pose
    print()
    for pair in POSE_PAIRS:
        part_a = pair[0]  # 0 (Head)
        part_b = pair[1]  # 1 (Neck)
        if points[part_a] and points[part_b]:
            print(f"[linked] {part_a} {points[part_a]} <=> {part_b} {points[part_b]}")
            cv2.line(frame, points[part_a], points[part_b], (0, 255, 0), 3)
            cv2.putText(frame, pose,(30, 100), font, 7,(0,0,0),9, cv2.LINE_AA)
            cv2.putText(frame, msg, (30, 200), font, 5,(0,0,0),5, cv2.LINE_AA)
            
        else:
            print(f"[not linked] {part_a} {points[part_a]} <=> {part_b} {points[part_b]}")

    cv2.imshow("output_keypoints_with_lines", frame)

# 웹캠에서 영상 가져와서 함수에 전달
# mode 선택받아 그에 따른 모델 사용
# detailed mode 일 때 10 timer 기능 포함
# timer start, restart, 종료 액션을 제어하는 키 입력
select=-1
TIMER=int(10)
cap=cv2.VideoCapture(0)
prev=time.time()
if cap.isOpened():
    while True:
        if select ==-1:
            img=cv2.imread('./imagedir/ballet.jpg', cv2.IMREAD_COLOR)
            img=cv2.resize(img, (1161,960), interpolation=cv2.INTER_LINEAR)
            font=cv2.FONT_HERSHEY_PLAIN
            cv2.putText(img, "select mode",(240, 350), font, 7,(0,0,0),9, cv2.LINE_AA)
            cv2.putText(img, "[1]: Real Time Mode",(320, 450), font, 3,(0,0,0),8, cv2.LINE_AA)
            cv2.putText(img, "[2]: Detailed Mode",(320, 550), font, 3,(0,0,0),8, cv2.LINE_AA)
            cv2.putText(img, "select mode",(240, 350), font, 7,(255,255,255),2, cv2.LINE_AA)
            cv2.putText(img, "[1]: Real Time Mode",(320, 450), font, 3,(255,255,255),1, cv2.LINE_AA)
            cv2.putText(img, "[2]: Detailed Mode",(320, 550), font, 3,(255,255,255),1, cv2.LINE_AA)
            cv2.imshow('output_keypoints_with_lines', img)
            while (1):
                if (cv2.waitKey(0)==ord('1')):
                    select=1
                    break
                elif (cv2.waitKey(0)==ord('2')):
                    select=2
                    break

        ret,frame=cap.read()
        if ret:
            if TIMER>=0 and select ==2:
                frame=cv2.flip(frame, 1)
                font=cv2.FONT_HERSHEY_PLAIN
                cv2.putText(frame, str(TIMER),(300, 300), font, 7,(255,255,255),6, cv2.LINE_AA)
                cv2.putText(frame, "Take a pose",(180, 200), font, 3,(255,255,255),3, cv2.LINE_AA)
                frame=cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                cv2.imshow('output_keypoints_with_lines',frame)
                cur=time.time()
                if cur-prev>1:
                    prev=cur
                    TIMER-=1
            elif TIMER < 0 and select==2:
                frame=cv2.flip(frame, 1)
                font=cv2.FONT_HERSHEY_PLAIN
                text_img=frame.copy()
                cv2.putText(text_img, "Please Wait :)",(180, 200), font, 3,(255,255,255),3, cv2.LINE_AA)
                frame=cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                text_img=cv2.resize(text_img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                cv2.imshow('output_keypoints_with_lines',text_img)
                cv2.waitKey(10)
                frame = output_keypoints(frame=frame, proto_file=protoFile_body_25, weights_file=weightsFile_body_25,
                                threshold=0.2, model_name="BODY_25", BODY_PARTS=BODY_PARTS_BODY_25)
                output_keypoints_with_lines(frame=frame, POSE_PAIRS=POSE_PAIRS_BODY_25)
                while (1):
                    if (cv2.waitKey(0)==ord('q')):
                        TIMER=10
                        break
            elif select==1:
                frame=cv2.flip(frame, 1)
                font=cv2.FONT_HERSHEY_PLAIN
                frame=cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                frame = output_keypoints(frame=frame, proto_file=protoFile_mpi_faster, weights_file=weightsFile_mpi,
                                threshold=0.2, model_name="MPI", BODY_PARTS=BODY_PARTS_MPI)
                output_keypoints_with_lines(frame=frame, POSE_PAIRS=POSE_PAIRS_MPI)

            if cv2.waitKey(1)==27:
                break
        else:
            print('no frame')
            break
else:
    print("can't open camera.")
cap.release()
cv2.destroyAllWindows()