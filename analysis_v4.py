"""
Add more landmarks for jump detection
"""
import os
import cv2
import glob
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import math
import pandas as pd
from pd_hz import pd_hz_hanshu

class Analysis():
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # 视频文件所在的根目录，毕竟是传过来的，都能理解

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        # 初始化姿势估计工具
        self.pose = self.mp_pose.Pose(model_complexity=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        # 创建一个姿势估计实例，设置模型复杂度为 2，检测置信度为 0.7，跟踪置信度为 0.7
        
    def detect_pose_landmarks(self, frame):
        """Detects pose landmarks in a given frame.
        检测给定帧中的人体姿势关键点

        Args:
            frame (np.ndarray): Video frame.
            输入视频帧（图像）

        Returns:
            Optional[np.ndarray]: Array of pose landmarks or None if no landmarks detected.
            返回世界坐标和普通坐标的姿势关键点，如果没有检测到，返回 None
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 将帧从 BGR 转换为 RGB 格式（MediaPipe 要求 RGB 格式输入）
        results = self.pose.process(rgb_frame)
        # 使用 MediaPipe 处理帧，获取人体关键点

        if results.pose_world_landmarks and results.pose_landmarks:
            world_landmarks = results.pose_world_landmarks.landmark
            landmarks =results.pose_landmarks.landmark
            return [world_landmarks, landmarks]

        return None, None
    
    def get_normal_vect_from_three_3d_points(self, p1, p2, p3):
        """
        get plane equation from 3 3d points
        该函数用于计算三维平面的法向量
        steps 具体实现步骤:
        1. Calculate vector p1p2 (as vector_1) and vector p1p3 (as vector_2)
            vector_1 = (x2 - x1, y2 - y1, z2 - z1) = (a1, b1, c1).
            vector_2 = (x3 - x1, y3 - y1, z3 - z1) = (a2, b2, c2).
        2. Get normal vector n of this plane by calculate outer product of vector p1p2 and vector p1p3
            vector_1 X vector_2 = (b1 * c2 - b2 * c1) i 
                        + (a2 * c1 - a1 * c2) j 
                        + (a1 * b2 - b1 * a2) k 
                    = ai + bj + ck.
        3. Get d and sub into ax + by + cz = d. Formula of d is d =  - a * x1 - b * y1 - c * z1
        """
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        x3, y3, z3 = p3

        a1 = x2 - x1
        b1 = y2 - y1
        c1 = z2 - z1
        a2 = x3 - x1
        b2 = y3 - y1
        c2 = z3 - z1

        a = (b1 * c2) - (b2 * c1)
        b = (a2 * c1) - (a1 * c2)
        c = (a1 * b2) - (b1 * a2)

        d = - a * x1 - b * y1 - c * z1

        normal_vector = np.array([a,b,c])
        # print(f"Normal Vector: {normal_vector}")      
        return normal_vector
    
    def calculate_angle_3d(self, norm_vect):
        """
        该函数计算法向量与水平面的夹角
        norm_vect: 输入的法向量
        This function is to calculate angle between left_foot_plane and horizontal plane.
        So, we set y as 1 since the horizontal plane's normal vector direction is pointing upwards.
        normal vector of the horizontal plane: (0, 1, 0).
        设 y 为 1 因为水平面的法向量方向是向上的
        水平面的法向量:(0,1,0)
        返回: 法向量与水平面的夹角，单位是度数
        """
        A1, B1, C1 = norm_vect
        A2, B2, C2 = 0, 1, 0
        numerator = (A1 * A2) + (B1 * B2) + (C1 * C2)
        denominator = math.sqrt(A1**2 + B1**2 + C1**2) * math.sqrt(A2**2 + B2**2 + C2**2)

        angle = math.acos(numerator/denominator)
        angle = round(math.degrees(angle), 2)
        # print(angle)
        return 90 - angle
    
    # 下蹲计算（函数名就是拼音）
    # p2 是顶点
    def xia_dun_ji_suan(self, p1, p2, p3):
        # 计算两个向量
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]])

        # 计算点积
        dot_product = np.dot(v1, v2)

        # 计算向量的模长
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        # 计算夹角（弧度）
        cos_theta = dot_product / (norm_v1 * norm_v2)

        # 通过反余弦函数得到夹角（度数）
        xia_dun = np.degrees(np.arccos(cos_theta))

        return xia_dun
    
    def l_r_ju_li(self, l_h, r_h):
        # 计算两点 (l_j = 左手, r_h = 右手) 之间的欧几里得距离
        distance = np.linalg.norm(np.array(l_h) - np.array(r_h))
        return distance
    
    def label(self, frame, landmarks, angle, io, peak, nose, avg_nose_height, frame_width, frame_height, norm_vect_io, xia_dun_angle, dis_lr_wrists):
        # ----- new landmarks ----- #
        left_shoulder = np.array([landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                  landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y])
        right_shoulder = np.array([landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                   landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
        left_hip = np.array([landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                             landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y])
        right_hip = np.array([landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x,
                              landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y])
        # ----- new landmarks ----- #

        cv2.circle(frame, (int(nose[0] * frame_width), int(nose[1] * frame_height)), 5, (0, 255, 0), -1)  # Nose
        cv2.circle(frame, (int(left_shoulder[0] * frame_width), int(left_shoulder[1] * frame_height)), 5, (0, 255, 0), -1)  # Left Shoulder
        cv2.circle(frame, (int(right_shoulder[0] * frame_width), int(right_shoulder[1] * frame_height)), 5, (0, 255, 0), -1)  # Right Shoulder
        cv2.circle(frame, (int(left_hip[0] * frame_width), int(left_hip[1] * frame_height)), 5, (0, 255, 0), -1)  # Left Hip
        cv2.circle(frame, (int(right_hip[0] * frame_width), int(right_hip[1] * frame_height)), 5, (0, 255, 0), -1)  # Right Hip

        legend_x, legend_y = 20, 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        # Draw legend background
        cv2.rectangle(frame, (legend_x, legend_y), (legend_x + 210, legend_y + 250), (255, 255, 255), -1)

        # Display data
        cv2.putText(frame, f'Angle: {angle:.2f}', (legend_x + 10, legend_y + 20), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(frame, f'IO: {io}', (legend_x + 10, legend_y + 40), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(frame, f'Peak: {peak}', (legend_x + 10, legend_y + 60), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(frame, f'Nose Height: {nose[1]:.2f}', (legend_x + 10, legend_y + 80), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(frame, f'Avg Nose Height: {avg_nose_height:.2f}', (legend_x + 10, legend_y + 100), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(frame, f'Left Shoulder: {left_shoulder[1]:.4f}', (legend_x + 10, legend_y + 120), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(frame, f'Right Shoulder: {right_shoulder[1]:.4f}', (legend_x + 10, legend_y + 140), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(frame, f'Left Hip: {left_hip[1]:.4f}', (legend_x + 10, legend_y + 160), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(frame, f'Right Hip: {right_hip[1]:.4f}', (legend_x + 10, legend_y + 180), font, font_scale, (0, 0, 0), thickness)
        
        avg_height = (round(left_shoulder[1], 4) + round(right_shoulder[1], 4) + round(left_hip[1], 4) + round(right_hip[1], 4))/4
        cv2.putText(frame, f'AVG Height: {avg_height:.4f}', (legend_x + 10, legend_y + 200), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(frame, f'Normal Vect IO: {norm_vect_io:.4f}', (legend_x + 10, legend_y + 220), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(frame, f'dis lr wrists: {dis_lr_wrists:.4f}', (legend_x + 10, legend_y + 240), font, font_scale, (0, 0, 0), thickness)

        return frame
    
    def add_peak_seq(self, upwards_peak, downwards_peak):
        """
        根据上下峰值判断峰值序列
        let:
        upwards peak as 1
        downwards peak as -1
        no peak as 0
        """
        if upwards_peak and not downwards_peak:
            return 1
        elif not upwards_peak and downwards_peak:
            return -1
        elif not upwards_peak and not downwards_peak:
            return 0
    
    def io_blade_estimation(self, angle):
        """
        根据角度估算姿态的内外倾斜
        angle : 输入的角度
        return : 根据角度返回 "inside"（内倾）、"outside"（外倾）或 "straight or blur"（直或模糊）
        """
        io = ""
        if 0 <= angle <= 90: # first quadrant
            if 9.9 > angle:
                io = "error"
            elif 80 >= angle:
                io = "inside"
            else:
                io = "straight or blur"
        elif -90 <= angle <= 0: #second quadrant
            if angle > -9.9:
                io = "error"
            elif angle > -80:
                io = "outside"
            else:
                io = "straight or blur"

        return io
    
    def norm_point_io_body(self, norm_vector, left_ankle):
        """
        计算身体某点与法向量的相对位置：
        norm_vector: 法向量
        left_ankle: 左脚踝坐标
        返回: 计算后的相对值
        """
        return round(left_ankle[0] - norm_vector[0], 4)
    
    def calculate_time_by_frame(self, fps, frame_count):
        """
        计算当前帧的时间戳：
        fps: 视频的帧率
        frame_count: 当前帧号
        返回: 该帧的时间戳（秒）
        """
        return frame_count / fps
    
    def calculate_velocity(self, y_previous, y_now, time):
        """
        计算物体的速度：
        y_previous: 前一时刻的 y 坐标
        y_now: 当前的 y 坐标
        time: 时间间隔
        返回: 速度
        """
        displacement = y_now - y_previous
        return displacement / time
    
    def save_data_to_excel(self, data, path):
        """
        将分析数据保存到 Excel 文件：
        data: 要保存的数据
        path: 保存路径
        df: 将数据转为 DataFrame 并保存到 Excel
        """
        # Convert the data to DataFrame
        df = pd.DataFrame({
            "Peak seq": data[0],
            "Angle Data": data[1],
            "IO": data[2],
            "Left Shoulder": data[3],
            "Right Shoulder": data[4],
            "Left Hip": data[5],
            "Left Foot Index": data[6],
            "Right Hip": data[7],
            "AVG Height": data[8],
            "dis lr wrists": data[9]
        })

        # Save the DataFrame to an Excel file
        df.to_excel(path, index=False)

    def save_img(self, frame, path):
        """
        将当前帧保存为图像文件：
        frame: 当前帧
        path: 保存路径
        """
        cv2.imwrite(path, frame)

    # 这边开始我自己写的，主要想解决视频太长的问题，就针对滑行的部分输出节点
    def detect_foot_lift_place(self, left_ankle_y, fps, frame_count, current_time, state):
        """
        检测脚是否抬起或放下，并打印时间节点。

        Args:
            left_ankle_y (float): 当前帧左脚踝的 y 坐标。
            fps (float): 视频的帧率。
            frame_count (int): 当前帧数。
            current_time (float): 当前时间戳（秒）。
            state (dict): 当前状态，包含 'is_lifted' 标识。

        Returns:
            dict: 更新后的状态。
        """
        # 设定一个静态阈值或者动态阈值，这里使用静态阈值示例
        # 可以根据需要调整阈值
        LIFT_THRESHOLD = 0.1  # 根据实际情况调整

        if 'avg_y' not in state:
            state['avg_y'] = left_ankle_y
            state['is_lifted'] = False
            return state

        # 平滑处理，更新平均 y 坐标
        alpha = 0.9
        state['avg_y'] = alpha * state['avg_y'] + (1 - alpha) * left_ankle_y

        # 判断脚是否抬起
        if left_ankle_y < state['avg_y'] - LIFT_THRESHOLD and not state['is_lifted']:
            # 脚被抬起
            print(f"[{current_time:.2f} s] 脚抬起")
            state['is_lifted'] = True
        elif left_ankle_y > state['avg_y'] + LIFT_THRESHOLD and state['is_lifted']:
            # 脚被放下
            print(f"[{current_time:.2f} s] 脚放下")
            state['is_lifted'] = False

        return state
    
    def process_videos_in_folder(self):
        """
        Processes all videos in the given folder.
            开始处理文件夹中的所有视频文件
        Args:
            root_folder (str): Path to the root folder containing videos.
        """
        video_files = glob.glob(os.path.join(self.root_dir, '**', '*.mp4'), recursive=True)
        total_videos = len(video_files)

        if total_videos == 0:
            print("No video files found.")
            return

        for video_path in tqdm(video_files, desc='Processing videos'):
            image_saved = 0 
            output_folder = os.path.join(os.path.dirname(video_path), "angle")
            os.makedirs(output_folder, exist_ok=True)
            output_video_path = os.path.join(output_folder, os.path.splitext(os.path.basename(video_path))[0] + "_angle.mp4")
            analysis_data_output_path = os.path.join(output_folder, os.path.splitext(os.path.basename(video_path))[0] + "_data.xlsx")

            cap = cv2.VideoCapture(video_path)

            # 检查视屏模糊
            if pd_hz_hanshu(video_path):
                continue

            # obtain width, height and fps of the video
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = 0
            fps = cap.get(cv2.CAP_PROP_FPS)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

            if not cap.isOpened():
                continue

            angle_3d_data = []
            io_blade = []

            previous_nose_height = 0 # Initialize the nose height
            nose_heights = []
            downwards_peak = False
            upwards_peak = False
            peak_value = 0.0
            peak_seq = []

            # ----- new landmarks list ----- #
            left_shoulders = []
            right_shoulders = []
            left_hips = []
            left_foot_indexes = []
            right_hips = []
            avg_heights = []
            xia_dun_angles = []
            dis_lr_wrists = []
            # velocities = [0.0]
            # 初始化检测脚抬起和放下的状态
            foot_state = {}
            # ----- new landmarks list ----- #

            while cap.isOpened():
                ret, frame = cap.read()
                frame_count += 1
                if not ret:
                    break

                world_3d_landmarks, landmarks = self.detect_pose_landmarks(frame)
                if world_3d_landmarks is not None:
                    # get body landmarks
                    left_ankle = np.array([world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].x,
                                           world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].y,
                                           world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].z])
                    left_hip_w = np.array([world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                                         world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y,
                                         world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].z])
                    left_knee = np.array([world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].x,
                                         world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].y,
                                         world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].z])
                    left_heel = np.array([world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL].x,
                                          world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL].y,
                                          world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL].z])
                    left_foot_index_w = np.array([world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x,
                                                world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y,
                                                world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].z])
                    left_wrist_w = np.array([world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].x,
                                             world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y,
                                             world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].z])
                    right_writst_w = np.array([world_3d_landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].x,
                                             world_3d_landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].y,
                                             world_3d_landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST].z])
                    nose = np.array([landmarks[self.mp_pose.PoseLandmark.NOSE].x,
                                     landmarks[self.mp_pose.PoseLandmark.NOSE].y])
                    nose_height = nose[1]
                    # ----- new landmarks ----- #
                    left_shoulder = np.array([landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                              landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y])
                    right_shoulder = np.array([landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                               landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y])
                    left_hip = np.array([landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                                         landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y])
                    left_foot_index = np.array([landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x,
                                         landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y])
                    right_hip = np.array([landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].x,
                                          landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP].y])
                    # ----- new landmarks ----- #
                    left_shoulders.append(round(left_shoulder[1], 4))
                    right_shoulders.append(round(right_shoulder[1], 4))
                    left_hips.append(round(left_hip[1], 4))
                    left_foot_indexes.append(round(left_foot_index[1], 4))
                    right_hips.append(round(right_hip[1], 4))
                    avg_height = (round(left_shoulder[1], 4) + round(right_shoulder[1], 4) + round(left_hip[1], 4) + round(right_hip[1], 4))/4
                    avg_heights.append(avg_height)

                    # calc normal vector of foot     
                    norm_vect = self.get_normal_vect_from_three_3d_points(left_ankle, left_heel, left_foot_index_w)
                    # calc angle
                    angle = self.calculate_angle_3d(norm_vect)
                    # estimate inside or outside blade
                    io = self.io_blade_estimation(angle)

                    # if io = "error" save the frame
                    # time_calctd = self.calculate_time_by_frame(fps, frame_count)
                    # if io == "error" and (3.75 <= time_calctd < 4.25):
                    #     print(time_calctd)
                    #     print(output_frame_path)
                    #     output_frame_path = os.path.join(error_folder_by_name, os.path.splitext(os.path.basename(video_path))[0] + "_{}.jpg".format(image_saved))
                    #     print(output_frame_path)
                    #     self.save_img(frame, output_frame_path)
                    #     image_saved += 1

                    norm_vect_io = self.norm_point_io_body(norm_vect, left_ankle)
                    # calc xia_dun angle
                    xia_dun_angle = self.xia_dun_ji_suan(left_hip_w, left_knee, left_ankle)
                    xia_dun_angles.append(xia_dun_angle)
                    dis_lr_wrist = self.l_r_ju_li(left_wrist_w, right_writst_w)
                    #print(dis_lr_wrist)
                    dis_lr_wrists.append(dis_lr_wrist)

                    # get peak value of nose on y axis
                    if previous_nose_height == 0: # first frame
                        previous_nose_height = nose_height 
                        nose_heights.append(nose_height)
                        avg_nose_height = sum(nose_heights)/len(nose_heights)
                        peak_seq.append(self.add_peak_seq(upwards_peak, downwards_peak))
                    elif round(avg_nose_height - nose_height, 3) >= 0.045: # estimate upwards peak
                        avg_nose_height = sum(nose_heights)/len(nose_heights)
                        peak_value = abs(avg_nose_height - nose_height)
                        nose_heights.append(nose_height)
                        previous_nose_height = nose_height
                        upwards_peak = True
                        peak_seq.append(self.add_peak_seq(upwards_peak, downwards_peak))
                    elif round(avg_nose_height - nose_height, 3) <= -0.045: # estimate downwards peak
                        avg_nose_height = sum(nose_heights)/len(nose_heights)
                        peak_value = abs(avg_nose_height - nose_height)
                        nose_heights.append(nose_height)
                        previous_nose_height = nose_height
                        downwards_peak = True
                        peak_seq.append(self.add_peak_seq(upwards_peak, downwards_peak))
                    else: # no peak
                        avg_nose_height = sum(nose_heights)/len(nose_heights)
                        peak_value = abs(avg_nose_height - nose_height)
                        nose_heights.append(nose_height)
                        previous_nose_height = nose_height
                        downwards_peak = False
                        upwards_peak = False
                        peak_seq.append(self.add_peak_seq(upwards_peak, downwards_peak))

                    # calc time now
                    # if frame_count > 1:
                    #     print(f"frame count: {frame_count}") 
                    #     print(f"velocities len: {len(velocities)}")
                    #     time_now = self.calculate_time_by_frame(fps, frame_count)
                    #     velocity = self.calculate_velocity(nose_heights[len(nose_heights)-2], nose_height, time_now)
                    #     velocities.append(velocity)
                        # print(f"time_now: {time_now}")
                        # print(f"velocity of nose: {velocity}")
                    current_time = self.calculate_time_by_frame(fps, frame_count)

                    # 检测脚抬起和放下
                    left_ankle_y = left_ankle[1]
                    foot_state = self.detect_foot_lift_place(left_ankle_y, fps, frame_count, current_time, foot_state)

                    # add angle data calculated before to angle_3d_data
                    angle_3d_data.append(angle)
                    # add io data calculated before to io_blade
                    io_blade.append(io)

                    # plot data into frame
                    frame = self.label(frame, landmarks, angle, io, self.add_peak_seq(upwards_peak, downwards_peak), nose, avg_nose_height, frame_width, frame_height, norm_vect_io, xia_dun_angle, dis_lr_wrist)

                out.write(frame)
            #print(len(peak_seq))
            #print(len(velocities))
            self.save_data_to_excel([peak_seq, angle_3d_data, io_blade, left_shoulders, right_shoulders, left_hips, left_foot_indexes, right_hips, avg_heights, dis_lr_wrists], analysis_data_output_path)  # save data to excel                 
            cap.release()
            out.release()