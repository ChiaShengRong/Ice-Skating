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

class Analysis():
    def __init__(self, root_dir):
        self.root_dir = root_dir

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(model_complexity=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        
    def detect_pose_landmarks(self, frame):
        """Detects pose landmarks in a given frame.

        Args:
            frame (np.ndarray): Video frame.

        Returns:
            Optional[np.ndarray]: Array of pose landmarks or None if no landmarks detected.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_world_landmarks and results.pose_landmarks:
            world_landmarks = results.pose_world_landmarks.landmark
            landmarks =results.pose_landmarks.landmark
            return [world_landmarks, landmarks]

        return None, None
    
    def get_normal_vect_from_three_3d_points(self, p1, p2, p3):
        """
        get plane equation from 3 3d points
        steps:
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
        This function is to calculate angle between left_foot_plane and horizontal plane.
        So, we set y as 1 since the horizontal plane's normal vector direction is pointing upwards.
        normal vector of the horizontal plane: (0, 1, 0). 
        """
        A1, B1, C1 = norm_vect
        A2, B2, C2 = 0, 1, 0
        numerator = (A1 * A2) + (B1 * B2) + (C1 * C2)
        denominator = math.sqrt(A1**2 + B1**2 + C1**2) * math.sqrt(A2**2 + B2**2 + C2**2)

        angle = math.acos(numerator/denominator)
        angle = round(math.degrees(angle), 2)
        # print(angle)
        return 90 - angle
    
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
        # 计算两点之间的欧几里得距离
        distance = np.linalg.norm(np.array(l_h) - np.array(r_h))
        return distance
    
    def label(self, frame, landmarks, angle, io, peak, nose, avg_nose_height, dist_hip, frame_width, frame_height,):
        # ----- new landmarks ----- #
        left_hip = np.array([landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].x,
                             landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y])
        # ----- new landmarks ----- #

        #cv2.circle(frame, (int(nose[0] * frame_width), int(nose[1] * frame_height)), 5, (0, 255, 0), -1)  # Nose
        #cv2.circle(frame, (int(left_hip[0] * frame_width), int(left_hip[1] * frame_height)), 5, (0, 255, 0), -1)  # Left Hip

        # Iterate over all the landmarks in the PoseLandmark enumeration
        for landmark in self.mp_pose.PoseLandmark:
            skip_landmarks = [
                                            self.mp_pose.PoseLandmark.LEFT_EYE_INNER,  # Left Eye (Inner)
                                            self.mp_pose.PoseLandmark.LEFT_EYE,        # Left Eye
                                            self.mp_pose.PoseLandmark.LEFT_EYE_OUTER,  # Left Eye (Outer)
                                            self.mp_pose.PoseLandmark.RIGHT_EYE_INNER, # Right Eye (Inner)
                                            self.mp_pose.PoseLandmark.RIGHT_EYE,       # Right Eye
                                            self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER, # Right Eye (Outer)
                                            self.mp_pose.PoseLandmark.LEFT_EAR,        # Left Ear
                                            self.mp_pose.PoseLandmark.RIGHT_EAR,       # Right Ear
                                            self.mp_pose.PoseLandmark.MOUTH_LEFT,      # Mouth (Left)
                                            self.mp_pose.PoseLandmark.MOUTH_RIGHT,  # Mouth (Right)
                                            self.mp_pose.PoseLandmark.RIGHT_PINKY,
                                            self.mp_pose.PoseLandmark.RIGHT_INDEX,
                                            self.mp_pose.PoseLandmark.RIGHT_THUMB,
                                            self.mp_pose.PoseLandmark.LEFT_PINKY,
                                            self.mp_pose.PoseLandmark.LEFT_INDEX,
                                            self.mp_pose.PoseLandmark.LEFT_THUMB,

                                            ]
            
            if landmark in skip_landmarks:
                continue

            # Get the x and y coordinates of the landmark
            x = landmarks[landmark].x
            y = landmarks[landmark].y

            # Convert normalized coordinates to pixel coordinates
            x_pixel = int(x * frame_width)
            y_pixel = int(y * frame_height)

            # Circle the landmark on the frame (Green color)
            cv2.circle(frame, (x_pixel, y_pixel), 5, (0, 255, 0), -1)


        legend_x, legend_y = 20, 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        # Draw legend background
        cv2.rectangle(frame, (legend_x, legend_y), (legend_x + 210, legend_y + 150), (255, 255, 255), -1)

        # Display data
        cv2.putText(frame, f'Angle: {angle:.2f}', (legend_x + 10, legend_y + 20), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(frame, f'IO: {io}', (legend_x + 10, legend_y + 40), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(frame, f'Peak: {peak}', (legend_x + 10, legend_y + 60), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(frame, f'Nose Height: {nose[1]:.2f}', (legend_x + 10, legend_y + 80), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(frame, f'Avg Nose Height: {avg_nose_height:.2f}', (legend_x + 10, legend_y + 100), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(frame, f'Left Hip: {left_hip[1]:.4f}', (legend_x + 10, legend_y + 120), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(frame, f'dist_hip: {dist_hip:.4f}', (legend_x + 10, legend_y + 140), font, font_scale, (0, 0, 0), thickness)

        return frame
    
    def add_peak_seq(self, upwards_peak, downwards_peak):
        """
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
        io = ""
        if 0 <= angle <= 90: # first quadrant
            if 9.9 > angle:
                io = "error"
            elif 80 >= angle:
                io = "inside"
            else:
                io = "error"
        elif -90 <= angle <= 0: #second quadrant
            if angle > -9.9:
                io = "error"
            elif angle > -80:
                io = "outside"
            else:
                io = "error"

        return io
    
    def norm_point_io_body(self, norm_vector, left_ankle):
        return round(left_ankle[0] - norm_vector[0], 4)
    
    def calculate_time_by_frame(self, fps, frame_count):
        return frame_count / fps
    
    def calculate_velocity(self, y_previous, y_now, time):
        displacement = y_now - y_previous
        return displacement / time
    
    def recognize_peak(self, nose_heights, nose_height):
        """
        To recognize if athelete now is at -1 (ready) or 0 (normal sliding) or 1 (jumping) state.
        """
        avg_nose_height = sum(nose_heights)/len(nose_heights)
        if round(avg_nose_height - nose_height, 3) >= 0.045: # estimate upwards peak
            peak_value = abs(avg_nose_height - nose_height)
            nose_heights.append(nose_height)
            upwards_peak = True
            downwards_peak = False
        elif round(avg_nose_height - nose_height, 3) <= -0.045: # estimate downwards peak
            peak_value = abs(avg_nose_height - nose_height)
            nose_heights.append(nose_height)
            upwards_peak = False
            downwards_peak = True
        else: # no peak
            peak_value = abs(avg_nose_height - nose_height)
            nose_heights.append(nose_height)
            downwards_peak = False
            upwards_peak = False
        return peak_value, upwards_peak, downwards_peak
    

    def recognize_ready(self, peak_seq, pre_lhip_height, left_hip_height):
        """
        Judging if the athelete is in ready state(-1) by the hips
        """
        dist = left_hip_height - pre_lhip_height
        if dist > 0.03:
            return -1, dist
        
        return peak_seq[-1], dist
    
    
    def save_data_to_excel(self, data, path):
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
        cv2.imwrite(path, frame)
    

    def process_videos_in_folder(self):
        """Processes all videos in the given folder.

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
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            os.makedirs(output_folder, exist_ok=True)
            error_folder_by_name = os.path.join(os.path.dirname(output_folder), video_name)
            #os.makedirs(error_folder_by_name, exist_ok=True)
            output_video_path = os.path.join(output_folder, os.path.splitext(os.path.basename(video_path))[0] + "_angle.mp4")
            analysis_data_output_path = os.path.join(output_folder, os.path.splitext(os.path.basename(video_path))[0] + "_data.xlsx")

            cap = cv2.VideoCapture(video_path)

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
            pre_lhip_height = 0 # Initialize the left hip height
            nose_heights = []
            left_hip_heights = []
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
                    left_knee_w = np.array([world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].x,
                                         world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].y,
                                         world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE].z])
                    left_heel_w = np.array([world_3d_landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL].x,
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
                    left_hip_height = left_hip[1]
                    # ----- new landmarks ----- #
                    left_shoulders.append(round(left_shoulder[1], 4))
                    right_shoulders.append(round(right_shoulder[1], 4))
                    left_hips.append(round(left_hip[1], 4))
                    left_foot_indexes.append(round(left_foot_index[1], 4))
                    right_hips.append(round(right_hip[1], 4))
                    avg_height = (round(left_shoulder[1], 4) + round(right_shoulder[1], 4) + round(left_hip[1], 4) + round(right_hip[1], 4))/4
                    avg_heights.append(avg_height)

                    # calc normal vector of foot     
                    norm_vect = self.get_normal_vect_from_three_3d_points(left_foot_index_w, left_ankle, left_heel_w)
                    # calc angle
                    angle = self.calculate_angle_3d(norm_vect)
                    # estimate inside or outside blade
                    io = self.io_blade_estimation(angle)

                    norm_vect_io = self.norm_point_io_body(norm_vect, left_ankle)
                    # calc xia_dun angle
                    xia_dun_angle = self.xia_dun_ji_suan(left_hip_w, left_knee_w, left_ankle)
                    xia_dun_angles.append(xia_dun_angle)
                    dis_lr_wrist = self.l_r_ju_li(left_wrist_w, right_writst_w)
                    #print(dis_lr_wrist)
                    dis_lr_wrists.append(dis_lr_wrist)

                    # get peak value of nose on y axis
                    if previous_nose_height == 0: # first frame
                        previous_nose_height = nose_height 
                        nose_heights.append(nose_height)
                        peak_seq.append(self.add_peak_seq(upwards_peak, downwards_peak))
                    else:
                        nose_heights.append(nose_height)
                        peak_value, upwards_peak, downwards_peak = self.recognize_peak(nose_heights, nose_height)
                        peak_seq.append(self.add_peak_seq(upwards_peak, downwards_peak))

                    # change the peak seq by left hip height
                    dist_hip = 0.0
                    if pre_lhip_height == 0: # first frame
                        pre_lhip_height = left_hip_height
                        left_hip_heights.append(left_hip_height)
                    else:
                        left_hip_heights.append(left_hip_height)
                        peak_seq[-1], dist_hip = self.recognize_ready(peak_seq, pre_lhip_height, left_hip_height)

                    # add angle data calculated before to angle_3d_data
                    angle_3d_data.append(angle)
                    # add io data calculated before to io_blade
                    io_blade.append(io)

                    # plot data into frame
                    avg_nose_height = sum(nose_heights)/len(nose_heights)
                    frame = self.label(frame, landmarks, angle, io, peak_seq[-1], nose, avg_nose_height, dist_hip, frame_width, frame_height)

                out.write(frame)
            #print(len(peak_seq))
            #print(len(velocities))
            self.save_data_to_excel([peak_seq, angle_3d_data, io_blade, left_shoulders, right_shoulders, left_hips, left_foot_indexes, right_hips, avg_heights, dis_lr_wrists], analysis_data_output_path)  # save data to excel                 
            cap.release()
            out.release()