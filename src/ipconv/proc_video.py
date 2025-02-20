import os
import cv2
import numpy as np
from mvextractor.videocap import VideoCap


class DecodedVideo:
    def __init__(self, video_path):
        self.video_path = video_path
        self.video_cap = VideoCap()

        if not self.video_cap.open(video_path):
            raise FileNotFoundError(f"Failed to open video file at {video_path}")

        self.frames = []
        self.frame_types = []
        self.motion_vectors = []
        self.residuals = []
        self.shape = None

        while True:
            success, frame, motion_vectors, frame_type, timestamp = self.video_cap.read()
            if not success:
                break  # End of video
            
            if self.shape is None:
                self.shape = frame.shape

            if frame_type in ["P", "B"]:
                predicted_frame = apply_motion_vectors(self.frames[-1], motion_vectors, self.shape)
                residual = frame - predicted_frame
                self.residuals.append(residual)
            else:
                residual = np.ones_like(frame) * 255

            self.frames.append(frame)
            self.frame_types.append(frame_type)
            self.motion_vectors.append(motion_vectors)
            self.residuals.append(residual)
        
        self.frame_count = len(self.frames)

    def __len__(self):
        return self.frame_count
    
    def __getitem__(self, idx):
        return (self.frames[idx], self.frame_types[idx], self.motion_vectors[idx], self.residuals[idx])
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __del__(self):
        self.video_cap.release()


def jpeg_sequence_to_h264(jpeg_sequence_dir, output_dir='./data', verbose=False):
    """
    Convert a sequence of JPEG images to H.264 video using ffmpeg.
    :param jpeg_sequence_dir: Directory containing the JPEG images.
    :return: Path to the H.264 video file.
    """
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    output_name = os.path.basename(jpeg_sequence_dir) + '.mp4'
    output_path = os.path.join(output_dir, output_name)

    # Use ffmpeg to convert the JPEG sequence to H.264 video
    # -r 30: Set the frame rate to 30 FPS
    # -f image2: Set the input format to image2
    # -i <input_pattern>: Input file pattern
    # -c:v libx264: Set the video codec to libx264
    # -crf 17: Set the Constant Rate Factor (CRF) to 17
    # -pix_fmt yuv420p: Set the pixel format to yuv420p
    # -y: Overwrite output files without asking
    # -g 30: Set the keyframe interval to 30
    # -bf 0: Disable B-frames
    ffmpeg_cmd = f'''ffmpeg
        -r 30 -f image2 -i {jpeg_sequence_dir}/%05d.jpg
        -c:v libx264 -crf 17 -pix_fmt yuv420p -y -g 30 -bf 0 
        -x264-params "partitions=none:mb-tree=0" 
        {output_path}
    ''' + (' -hide_banner -loglevel error' if not verbose else '')
    
    os.system(ffmpeg_cmd)

    return output_path


def apply_motion_vectors(prev_frame, motion_vectors, frame_shape):
    """
    Motion Vector를 사용하여 이전 프레임을 기반으로 예측된 프레임을 생성.
    - OpenCV `cv2.remap()`을 활용하여 Bilinear Interpolation을 빠르게 수행.
    """
    h_max, w_max = frame_shape[:2]  # frame_shape = (height, width, channels)
    
    # Initialize the map_x and map_y arrays
    map_x = np.tile(np.arange(w_max, dtype=np.float32), (h_max, 1))
    map_y = np.tile(np.arange(h_max, dtype=np.float32).reshape(-1, 1), (1, w_max))

    # Apply motion vectors to create a new coordinate map
    for mv in motion_vectors:
        source, w, h, src_x, src_y, dst_x, dst_y, motion_x, motion_y, scale = mv

        if scale > 0:
            motion_subx = (motion_x % scale) / scale  # Fractional part
            motion_suby = (motion_y % scale) / scale
            motion_x = motion_x // scale  # Integer part
            motion_y = motion_y // scale

        # Destination block coordinates (always guaranteed to be non-negative)
        dst_x1 = max(0, dst_x - w // 2)
        dst_y1 = max(0, dst_y - h // 2)
        dst_x2 = min(w_max, dst_x1 + w)
        dst_y2 = min(h_max, dst_y1 + h)

        # Reference block coordinates (always guaranteed to be non-negative)
        ref_x1 = max(0, min(w_max - 1, dst_x + motion_x - w // 2 + motion_subx))
        ref_y1 = max(0, min(h_max - 1, dst_y + motion_y - h // 2 + motion_suby))
        ref_x2 = min(w_max, ref_x1 + w)
        ref_y2 = min(h_max, ref_y1 + h)

        # Adjust block sizes to prevent negative values
        block_w = max(1, min(ref_x2 - ref_x1, dst_x2 - dst_x1))
        block_h = max(1, min(ref_y2 - ref_y1, dst_y2 - dst_y1))

        # Clip negative values when using np.linspace()
        if dst_x2 > dst_x1 and dst_y2 > dst_y1:
            map_x[dst_y1:dst_y2, dst_x1:dst_x2] = np.clip(
                np.linspace(ref_x1, ref_x2, max(1, dst_x2 - dst_x1), endpoint=False), 0, w_max - 1
            )
            map_y[dst_y1:dst_y2, dst_x1:dst_x2] = np.clip(
                np.linspace(ref_y1, ref_y2, max(1, dst_y2 - dst_y1), endpoint=False).reshape(-1, 1), 0, h_max - 1
            )

    # Apply interpolation using OpenCV remap (Bilinear Interpolation)
    predicted_frame = cv2.remap(prev_frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    return predicted_frame


if __name__ == "__main__":
    import sys

    # DEMO 1: Convert a sequence of JPEG images to H.264 video
    jpeg_sequence_dir = '/data/DAVIS/JPEGImages/480p/bear'
    h264_video_path = jpeg_sequence_to_h264(jpeg_sequence_dir)
    print(f'H.264 video saved to {h264_video_path}')

    # DEMO 2: Load a decoded video
    decoded_video = DecodedVideo(h264_video_path)
    print(f'Loaded video with {len(decoded_video)} frames')

    # DEMO 3: Iterate over the frames of the decoded video
    for idx, (frame, frame_type, motion_vectors, residual) in enumerate(decoded_video):
        print(f'Frame {idx}: {frame_type} frame with shape {frame.shape} | {motion_vectors.shape} motion vectors')

    sys.exit(0)