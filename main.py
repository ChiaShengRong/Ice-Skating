import time
from analysis_v4 import Analysis

if __name__ == "__main__":
    root_dir = "C:\\Users\\khaw\\Desktop\\花滑视频\\Senior Women - SP1\\C0056"

    ja = Analysis(root_dir)

    start_time = time.time()  

    ja.process_videos_in_folder()

    end_time = time.time()
    # Calculate processing time
    total_process_time = end_time - start_time
    print(f"Total process time: {total_process_time:.2f} s")