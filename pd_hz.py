import cv2

def check_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

def detect_blur_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def check_video_blur(video_path):
    cap = cv2.VideoCapture(video_path)
    blur_threshold = 30  # 可根据需要调整
    frame_count = 0
    blurred_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        blur_value = detect_blur_frame(frame)
        if blur_value < blur_threshold:
            blurred_frames += 1
    cap.release()
    blur_ratio = blurred_frames / frame_count
    return blur_ratio

def pd_hz_hanshu(video_path):
    # 检查视频分辨率
    width, height = check_resolution(video_path)
    print(f"分辨率：{width} x {height}")
    resolution_threshold = (640, 360)  # 可以根据需要设置分辨率标准

    if width < resolution_threshold[0] or height < resolution_threshold[1]:
        print("分辨率太低")
        return 1  # 返回 1 表示分辨率不合格

    # 检查视频模糊程度
    blur_ratio = check_video_blur(video_path)
    print(f"模糊程度：{blur_ratio}")
    blur_threshold = 0.5  # 模糊帧比例阈值

    if blur_ratio > blur_threshold:
        print("画质太低")
        return 1  # 返回 1 表示画质不合格
    else:
        return 0  # 返回 0 表示可以继续处理