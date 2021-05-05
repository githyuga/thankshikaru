import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
import cv2
import copy
import random

nose = 0
l_eye = 1
r_eye = 2
l_ear = 3
r_ear = 4
l_shoulder = 5
r_shoulder = 6
l_elbow = 7
r_elbow = 8
l_hand = 9
r_hand = 10
l_waist = 11
r_waist = 12
l_knee = 13
r_knee = 14
l_ankle = 15
r_ankle = 16
keypoint_name = []

def keypoints_line_draw(img, person_pose, start_parts, end_parts, key_point_score, line_color):
    if keypoint_score[start_parts] < 0.8:
        return img
    if keypoint_score[end_parts] < 0.8:
        return img
    key_point_start = person_pose[start_parts]
    key_point_end = person_pose[end_parts]
    key_point_start_x = key_point_start[1]
    key_point_start_y = key_point_start[0]
    key_point_end_x = key_point_end[1]
    key_point_end_y = key_point_end[0]
    cv2.line(img, (key_point_start_x, key_point_start_y), (key_point_end_x, key_point_end_y), line_color, thickness=4, lineType=cv2.LINE_AA)
    return img

def cv2pil(imgCV):
    ''' OpenCV型 -> PIL型 '''
    imgPIL = Image.fromarray(imgCV)
    return imgPIL
# from chainercv.visualizations import vis_bbox, vis_point

image_path = "/home/ono/Downloads/mp_20180404-020940226_96ypu.jpg"
# image_path = "/home/ono/Downloads/ajsdjbcnkja.jpeg"#katakumi
# image_path = "/home/ono/Downloads/mp_20180404-020940226_96ypu.jpg"

# image = Image.open(image_path).convert('RGB')
# VideoCaptureのインスタンスを作成(引数でカメラを選択できる)
cap = cv2.VideoCapture(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model = model.to(device)
model.eval()
frame_index = 0
parts_list = [[39, 100], [39, 108], [39, 87], [39, 120], [39, 77], [74, 134], [74, 60], 
[113, 141], [113, 57], [142, 138], [142, 56], [143, 117], [143, 82], [193, 114], [193, 83], [237, 116], [237, 82]]
draw_order = [[0,1], [0,2], [2, 4], [1, 3], [6, 5], [6, 8], [5, 7], [8, 10], [7, 9], [11, 12], [12, 14], [11, 13], [14, 16], [13, 15]]
while True:
    frame_index = frame_index + 1
    save_name = str(frame_index).zfill(5)
    save_path = "/home/ono/Pictures/{}.png".format(save_name)
    # VideoCaptureから1フレーム読み込む
    ret, frame = cap.read() # 戻り値のframeがimg
    fig_person_path = "/home/ono/Downloads/istockphoto-823826708-1024x1024.jpg"
    fig_person = cv2.imread(fig_person_path)
    if type(frame) is np.ndarray:
        image_tensor = torchvision.transforms.functional.to_tensor(frame)
        x = [image_tensor.to(device)]
        prediction = model(x)[0]
        keypoints_np = prediction['keypoints'].to(torch.int16).cpu().numpy()
        bboxes_np = prediction['boxes'].to(torch.int16).cpu().numpy()
        labels_np = prediction['labels'].byte().cpu().numpy()
        scores_np = prediction['scores'].cpu().detach().numpy()
        keypoints_scores_np = prediction['keypoints_scores'].cpu().detach().numpy()
        bboxes = []
        labels = []
        scores = []
        keypoints = []
        keypoints_score = []

        for i, bbox in enumerate(bboxes_np):
            score = scores_np[i]
            if score < 0.8:
                continue
            label = labels_np[i]
            keypoint = keypoints_np[i]
            keypoint_score = keypoints_scores_np[i]
            bboxes.append([bbox[1], bbox[0], bbox[3], bbox[2]])
            labels.append(label - 1)
            scores.append(score)
            keypoints.append(keypoint)
            keypoints_score.append(keypoint_score)

        bboxes = np.array(bboxes)
        labels = np.array(labels)
        scores = np.array(scores)
        keypoints = np.array(keypoints)
        keypoints_score = np.array(keypoints_score)
        if keypoints.size == 0:
            cv2.imshow('Edited Frame', frame)
            # キー入力を1ms待って、keyが「q」だったらbreak
            key = cv2.waitKey(1)&0xff
            if key == ord('q'):
                break
            if key == ord('k'):
                cv2.imwrite(save_path, frame)  
            continue
        points = np.dstack([keypoints[:, :, 1], keypoints[:, :, 0]])
        # print(points[0])
        for box_num, bbox in enumerate(bboxes):
            score = scores[box_num]
            score_text = "score" + " : " + str(score)
            left_box_point = (bbox[1], bbox[0])
            right_box_point = (bbox[3], bbox[2])
            frame = cv2.rectangle(frame, right_box_point, left_box_point, (150, 150, 0), thickness=5)
            cv2.putText(frame, score_text, left_box_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), thickness=1)
        for person_num, person_pose in enumerate(points):
            b_line_color = random.randrange(255, 256)
            g_line_color = random.randrange(1)
            r_line_color = random.randrange(1)
            line_color = (b_line_color, g_line_color, r_line_color)
            b_circle_color = random.randrange(1)
            g_circle_color = random.randrange(1)
            r_circle_color = random.randrange(255, 256)
            circle_color = (b_circle_color, g_circle_color, r_circle_color)
            keypoint_score = keypoints_score[person_num]
            for line_num in draw_order:
                frame = keypoints_line_draw(frame, person_pose, line_num[0], line_num[1], keypoint_score, line_color)
                fig_person  = keypoints_line_draw(fig_person, parts_list, line_num[0], line_num[1], keypoint_score, line_color)
            for key_num, point in enumerate(person_pose):
                if keypoints_score[person_num][key_num] < 0.8:
                    continue
                fig_point = parts_list[key_num]
                key_num = str(key_num+1)
                point_y = point[0]
                point_x = point[1]
                # cv2.putText(frame, key_num, (point_x, point_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), thickness=2)
                cv2.circle(frame, (point_x, point_y), 5, circle_color, thickness=-1, lineType=cv2.LINE_8, shift=0)
                cv2.circle(fig_person, (fig_point[1], fig_point[0]), 5, circle_color, thickness=-1, lineType=cv2.LINE_8, shift=0)
            # cv2.imshow('image',frame)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
                # 加工した画像を表示
        cv2.imshow('Edited Frame', frame)
        # cv2.imshow('fig_person', fig_person)
        # キー入力を1ms待って、keyが「q」だったらbreak
        key = cv2.waitKey(1)&0xff
        if key == ord('q'):
            break
        if key == ord('k'):
            cv2.imwrite(save_path, frame)     
    else:
        pass
    

    # キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()
