import torch
import cv2

# Model​
model = torch.hub.load("ultralytics/yolov5", "yolov5m")

# Video capture
cap = cv2.VideoCapture(0)

# TODO: Loop for camera frames
while True :
    
# # Read frame (BGR to RGB)
    ret, frame = cap.read()
    
# # TODO: break the loop on error
    if not ret:
        break
# # TODO: break the loop on error

# 추론 실행 (BGR -> RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)

    # TODO: Boudning box 그리기
    for i, obj in enumerate(results.xyxy[0]):
        obj_info = list(map(int, obj))

        # TODO: 인식결과를 표시하기 위한 좌표를 얻음       
        # obj_info =( 0 = x1, 1 = y1, 2 = x2, 3 = y2, 4 = confidence(정확도), 5 = class(물체가 무엇인지?)
        # x1,y1 = 왼쪽 위박스, x2, y2 = 오른쪽 아래박스
        # 그래서 둘이 합치면 네모난 박스안에서 인심한 물체를 설정가능 
        x1, y1, x2, y2 = obj_info[0], obj_info[1], obj_info[2], obj_info[3]

        class_id = obj_info[5]

        class_name = model.names[class_id]

        # TODO: 인식된 정확도(confidence)와 클래스를 label로 구성
        confidence= obj[4]

        label = f"{class_name} {confidence:.2f}"
        # TODO: OpenCV를 이용해서 해당 좌표에 사각형과 text를 출력
        # cv2.rectangle= 프레임에서 찾은 좌표값을 네모난 박스로 그려줌. 그리고 뒤에 0,0,0은 그박스에 색상을 정함. 맨뒤는 박스의 두께를 정함
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.putTest= 프레임에서 찾은 좌표값에 라벨링을 함.(위에 라벨을 플로팅하여 클레스 네임과 정확도를 보여줌)
        # 세번째는 폰트를 어떻게 표현할지 보여줌 4번째는 폰트 크기 비율 뒤에는 색상을 정하고 그에 맞는 두께를 설정.
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


        print(f"Object {i}: {model.names[obj_info[5]]}")

    # TODO: 화면 표시
    cv2.imshow("win", cv2.resize(frame, (1280, 600)))
    # TODO: 종료를 위한 key 처리
    if cv2.waitKey(1) & 0xff == 27: #27 : ESC keyy
        break
cap.release()
cv2.destroyAllWindows()
