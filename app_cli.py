import sys
import asyncio
import argparse
import logging
import tempfile
import os
import torch
import torchaudio
import numpy as np
import cv2
import gradio as gr
from diffusers import DDIMScheduler, AutoencoderKL
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature
from omegaconf import OmegaConf
from moviepy.editor import VideoFileClip, ImageSequenceClip
from accelerate.utils import set_seed
from diffusers.utils.import_utils import is_xformers_available
from utils.retinaface import RetinaFace
from utils.face_alignment import get_cropped_head_256
from faceID.faceID import FaceRecognition
from enhancers.GFPGAN.GFPGAN import GFPGAN
import subprocess
from tqdm import tqdm
import onnxruntime

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lấy đường dẫn thư mục chứa script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Thiết lập cấu hình và checkpoint mặc định từ biến môi trường
DEFAULT_CONFIG_PATH = os.getenv("DEFAULT_CONFIG_PATH", os.path.join(script_dir, "configs/unet/second_stage.yaml"))
DEFAULT_CHECKPOINT_PATH = os.getenv("DEFAULT_CHECKPOINT_PATH", os.path.join(script_dir, "checkpoints", "unet.pt"))

# Tải cấu hình
config = OmegaConf.load(DEFAULT_CONFIG_PATH)

# Kiểm tra và thiết lập giá trị mặc định cho fps nếu không có trong cấu hình
if 'data' not in config:
    config.data = OmegaConf.create()
if 'fps' not in config.data:
    config.data.fps = 25  # Giá trị mặc định cho fps

# Tải scheduler
scheduler = DDIMScheduler.from_pretrained("configs", config_file="scheduler_config.json")

# Xác định đường dẫn mô hình Whisper dựa trên cross_attention_dim
if config.model.cross_attention_dim == 768:
    whisper_model_path = os.path.join(script_dir, "checkpoints", "whisper", "small.pt")
elif config.model.cross_attention_dim == 384:
    whisper_model_path = os.path.join(script_dir, "checkpoints", "whisper", "tiny.pt")
else:
    raise NotImplementedError("cross_attention_dim phải là 768 hoặc 384")

# Tải bộ mã hóa âm thanh
audio_encoder = Audio2Feature(model_path=whisper_model_path, device="cuda", num_frames=config.data.num_frames)

# Tải VAE
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
vae.config.scaling_factor = 0.18215
vae.config.shift_factor = 0

# Cache pipeline để tái sử dụng
pipeline_cache = {}

# Hàm tải mô hình UNet từ checkpoint
def load_unet(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} không tồn tại.")
    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        checkpoint_path,
        device="cpu",
    )
    unet = unet.to(dtype=torch.float16)
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()
    return unet

# Hàm tạo pipeline
def create_pipeline(unet):
    return LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to("cuda")

# Hàm điều chỉnh độ dài video theo âm thanh bằng cách lặp lại frames
def adjust_video_length_to_audio(video_frames, audio_duration, fps=25):
    video_duration = len(video_frames) / fps
    if audio_duration > video_duration:
        frames_needed = int(audio_duration * fps)
        current_frames = len(video_frames)
        repeat_times = frames_needed // current_frames
        remainder = frames_needed % current_frames
        adjusted_frames = video_frames * repeat_times + video_frames[:remainder]
    elif audio_duration < video_duration:
        # Trim frames to match audio duration
        frames_needed = int(audio_duration * fps)
        adjusted_frames = video_frames[:frames_needed]
    else:
        # No adjustment needed
        adjusted_frames = video_frames
    return adjusted_frames

# Hàm tiền xử lý video (giảm độ phân giải nếu cần)
def preprocess_video(video_path):
    # Get video properties
    clip = VideoFileClip(video_path)
    width = clip.w
    height = clip.h
    file_size = os.path.getsize(video_path)
    clip.close()

    # Check if compression is needed
    if width > 2048 and file_size > 500 * 1024 * 1024:
        logger.info("Video exceeds 2K resolution and 500MB size. Compressing...")
        # Calculate scale factor based on the longer side
        scale_factor = 1280 / max(width, height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        preprocessed_path = video_path.replace(".mp4", "_compressed.mp4")
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', video_path,
                '-vf', f'scale={new_width}:{new_height}',
                '-crf', '23', '-r', str(clip.fps),
                '-c:v', 'libx264', preprocessed_path
            ], check=True)
            return preprocessed_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Error occurred during video compression: {e}")
            return video_path
    else:
        logger.info("Video is within acceptable resolution and size. No compression needed.")
        return video_path

# Face detection model initialization
detector = RetinaFace(
    os.path.join(script_dir, "utils", "scrfd_2.5g_bnkps.onnx"),
    provider=[
        ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
        "CPUExecutionProvider"
    ],
    session_options=None
)

# Face recognition model initialization
recognition = FaceRecognition(os.path.join(script_dir, 'faceID', 'recognition.onnx'))

# Load the specified enhancer model
def load_enhancer(enhancer_name, device):
    if enhancer_name == 'gfpgan':
        return GFPGAN(model_path=os.path.join(script_dir, "enhancers", "GFPGAN", "GFPGANv1.4.onnx"), device=device)
    else:
        raise ValueError(f"Unknown enhancer: {enhancer_name}")

# Process a batch of frames
def process_batch(frame_buffer, enhancer, face_mask, out, frame_width, frame_height):
    frames, aligned_faces, mats = zip(*frame_buffer)
    enhanced_faces = enhancer.enhance_batch(aligned_faces)
    for frame, aligned_face, mat, enhanced_face in zip(frames, aligned_faces, mats, enhanced_faces):
        enhanced_face_resized = cv2.resize(enhanced_face, (aligned_face.shape[1], aligned_face.shape[0]))
        face_mask_resized = cv2.resize(face_mask, (enhanced_face_resized.shape[1], enhanced_face_resized.shape[0]))
        blended_face = (face_mask_resized * enhanced_face_resized + (1 - face_mask_resized) * aligned_face).astype(np.uint8)
        mat_rev = cv2.invertAffineTransform(mat)
        dealigned_face = cv2.warpAffine(blended_face, mat_rev, (frame_width, frame_height))
        mask = cv2.warpAffine(face_mask_resized, mat_rev, (frame_width, frame_height))
        final_frame = (mask * dealigned_face + (1 - mask) * frame).astype(np.uint8)
        out.write(final_frame)
    return

# Enhance the video
def enhance_video(video_path, enhancer_name, output_path=None):
    device = 'cpu'
    if onnxruntime.get_device() == 'GPU':
        device = 'cuda'
    print(f"Running on {device}")
    enhancer = load_enhancer(enhancer_name, device)
    video_stream = cv2.VideoCapture(video_path)
    if not video_stream.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    if output_path is None:
        output_path = os.path.splitext(video_path)[0] + f'_enhanced_{enhancer_name}.mp4'
    temp_video_path = output_path.replace('.', '_temp.')
    out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    face_mask = np.zeros((256, 256), dtype=np.uint8)
    face_mask = cv2.rectangle(face_mask, (66, 69), (190, 240), (255, 255, 255), -1)
    face_mask = cv2.GaussianBlur(face_mask.astype(np.uint8), (19, 19), cv2.BORDER_DEFAULT)
    face_mask = cv2.cvtColor(face_mask, cv2.COLOR_GRAY2RGB)
    face_mask = face_mask / 255
    batch_size = 1
    frame_buffer = []
    for _ in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = video_stream.read()
        if not ret:
            break
        bboxes, kpss = detector.detect(frame, input_size=(320, 320), det_thresh=0.3)
        if len(kpss) == 0:
            out.write(frame)
            continue
        aligned_face, mat = get_cropped_head_256(frame, kpss[0], size=256, scale=1.0)
        frame_buffer.append((frame, aligned_face, mat))
        if len(frame_buffer) >= batch_size:
            process_batch(frame_buffer, enhancer, face_mask, out, frame_width, frame_height)
            frame_buffer = []
    if frame_buffer:
        process_batch(frame_buffer, enhancer, face_mask, out, frame_width, frame_height)
    video_stream.release()
    out.release()
    print(f"Enhanced video frames saved to {temp_video_path}")
    audio_path = os.path.splitext(output_path)[0] + '.aac'
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'aac', '-b:a', '192k', audio_path], check=True)
    subprocess.run(['ffmpeg', '-y', '-i', temp_video_path, '-i', audio_path, '-c:v', 'libx264', '-crf', '23', '-preset', 'medium', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', output_path], check=True)
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    if os.path.exists(audio_path):
        os.remove(audio_path)
    print(f"Enhanced video with original audio saved to {output_path}")
    print(f"Original video remains at {video_path}")

# Hàm thực hiện inference với thanh tiến trình
async def run_inference(video_path, audio_path, guidance_scale, seed, checkpoint_path, progress=gr.Progress()):
    try:
        global pipeline_cache
        success_flag = False  # Cờ để kiểm tra quá trình có thành công hay không

        # Kiểm tra giá trị hợp lệ
        if guidance_scale < 0.1 or guidance_scale > 5.0:
            raise ValueError("Hệ số hướng dẫn phải nằm trong khoảng từ 0.1 đến 5.0")
        if seed < 0:
            raise ValueError("Seed không được là số âm")

        # Kiểm tra định dạng video và âm thanh
        if not video_path.lower().endswith(".mp4"):
            raise ValueError("Video phải có định dạng .mp4")
        if not audio_path.lower().endswith(".wav"):
            raise ValueError("Âm thanh phải có định dạng .wav")

        # Kiểm tra kích thước file
        if os.path.getsize(video_path) / (1024 * 1024) > 500:
            raise ValueError("Kích thước video không được vượt quá 500MB")
        if os.path.getsize(audio_path) / (1024 * 1024) > 50:
            raise ValueError("Kích thước âm thanh không được vượt quá 50MB")

        # Tiền xử lý video (giảm độ phân giải nếu cần)
        preprocessed_video_path = preprocess_video(video_path)

        # Đọc video và âm thanh
        video_clip = VideoFileClip(preprocessed_video_path)
        video_frames = [frame for frame in video_clip.iter_frames()]
        video_clip.close()

        waveform, sample_rate = torchaudio.load(audio_path)
        audio_duration = waveform.shape[1] / sample_rate

        # Điều chỉnh độ dài video theo âm thanh
        video_frames = adjust_video_length_to_audio(video_frames, audio_duration, fps=config.data.fps)

        # Lưu video đã điều chỉnh thành file tạm
        adjusted_video_path = os.path.join(tempfile.gettempdir(), "adjusted_video.mp4")
        video_clip = ImageSequenceClip(video_frames, fps=config.data.fps)
        video_clip.write_videofile(adjusted_video_path, codec='libx264', audio=False)

        # Tái khởi tạo pipeline nếu checkpoint thay đổi
        if checkpoint_path not in pipeline_cache:
            logger.info(f"Tải lại pipeline với checkpoint mới: {checkpoint_path}")
            pipeline_cache[checkpoint_path] = create_pipeline(load_unet(checkpoint_path))
        pipeline = pipeline_cache[checkpoint_path]

        # Thiết lập seed
        if seed != -1:
            set_seed(seed)
        else:
            torch.seed()
        logger.info(f"Seed khởi tạo: {torch.initial_seed()}")

        # Thực hiện inference
        output_video_path = os.path.join(tempfile.gettempdir(), "output_video.mp4")
        progress(0.5, desc="Bước 1/2: Đang xử lý video và âm thanh...")
        pipeline(
            video_path=adjusted_video_path,
            audio_path=audio_path,
            video_out_path=output_video_path,
            video_mask_path=output_video_path.replace(".mp4", "_mask.mp4"),
            num_frames=config.data.num_frames,
            num_inference_steps=config.run.inference_steps,
            guidance_scale=guidance_scale,
            weight_dtype=torch.float16,
            width=config.data.resolution,
            height=config.data.resolution,
        )

        # Enhance the output video using GFPGAN
        enhanced_output_path = output_video_path.replace(".mp4", "_enhanced.mp4")
        enhance_video(output_video_path, 'gfpgan', enhanced_output_path)

        # Đánh dấu quá trình thành công
        success_flag = True

        # Trả về video đầu ra và thông báo thành công
        progress(1.0, desc="Bước 2/2: Hoàn tất!")
        return enhanced_output_path, "Quá trình đồng bộ hóa và nâng cấp video hoàn tất thành công."
    except Exception as e:
        logger.error(f"Lỗi xảy ra: {str(e)}", exc_info=True)
        return None, f"Lỗi: {str(e)}"
    finally:
        # Giải phóng bộ nhớ GPU
        torch.cuda.empty_cache()
        # Dọn dẹp file tạm thời nếu quá trình không thành công
        if not success_flag:
            if 'output_video_path' in locals() and os.path.exists(output_video_path):
                os.remove(output_video_path)
            if 'preprocessed_video_path' in locals() and os.path.exists(preprocessed_video_path):
                os.remove(preprocessed_video_path)
            if 'adjusted_video_path' in locals() and os.path.exists(adjusted_video_path):
                os.remove(adjusted_video_path)

# Hàm chính để chạy từ dòng lệnh
def main(config, args):
    try:
        # Load the UNet model
        unet = load_unet(args.inference_ckpt_path)

        # Create the pipeline
        pipeline = create_pipeline(unet)

        # Preprocess video if necessary
        preprocessed_video_path = preprocess_video(args.video_path)

        # Load video and audio
        video_clip = VideoFileClip(preprocessed_video_path)
        video_frames = [frame for frame in video_clip.iter_frames()]
        video_clip.close()

        waveform, sample_rate = torchaudio.load(args.audio_path)
        audio_duration = waveform.shape[1] / sample_rate

        # Adjust video length to match audio
        video_frames = adjust_video_length_to_audio(video_frames, audio_duration, fps=config.data.fps)

        # Write adjusted video to a temporary file
        adjusted_video_path = os.path.join(tempfile.gettempdir(), "adjusted_video.mp4")
        video_clip = ImageSequenceClip(video_frames, fps=config.data.fps)
        video_clip.write_videofile(adjusted_video_path, codec='libx264', audio=False)

        # Perform inference
        output_video_path = args.video_out_path
        pipeline(
            video_path=adjusted_video_path,
            audio_path=args.audio_path,
            video_out_path=output_video_path,
            video_mask_path=output_video_path.replace(".mp4", "_mask.mp4"),
            num_frames=config.data.num_frames,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            weight_dtype=torch.float16,
            width=config.data.resolution,
            height=config.data.resolution,
        )

        # Enhance the output video using GFPGAN by default
        enhanced_output_path = output_video_path.replace(".mp4", "_enhanced.mp4")
        enhance_video(output_video_path, 'gfpgan', enhanced_output_path)

        # Clean up temporary files
        os.remove(adjusted_video_path)
        os.remove(output_video_path)

        print(f"Enhanced video saved to {enhanced_output_path}")
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        sys.exit(1)

# Giao diện Gradio
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("<h1>Ứng dụng Đồng bộ Hóa Video và Âm thanh</h1>")
        gr.Markdown("Tải lên video và âm thanh để đồng bộ hóa. Video đầu ra sẽ có độ dài bằng với âm thanh đầu vào.")

        # Các thành phần nhập liệu
        with gr.Row():
            video_input = gr.Video(label="Video đầu vào")
            audio_input = gr.Audio(label="Âm thanh đầu vào", type="filepath")
        with gr.Row():
            guidance_scale = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="Hệ số hướng dẫn (Guidance Scale)")
            seed = gr.Number(value=1247, label="Seed (Số ngẫu nhiên)")
        with gr.Row():
            checkpoint_dropdown = gr.Dropdown(
                choices=["checkpoints/unet.pt", "checkpoints/other_checkpoint.pt"],
                value="checkpoints/unet.pt",
                label="Chọn Checkpoint"
            )

        # Nút chạy inference
        run_button = gr.Button("Bắt đầu đồng bộ hóa")

        # Các thành phần đầu ra
        output_video = gr.Video(label="Video đầu ra")
        message_box = gr.Textbox(label="Thông báo", interactive=False)

        # Kết nối nút với hàm inference
        run_button.click(
            fn=run_inference,
            inputs=[video_input, audio_input, guidance_scale, seed, checkpoint_dropdown],
            outputs=[output_video, message_box]
        )

    # Khởi chạy ứng dụng với chế độ share=True
    # Thêm server_name và server_port vào hàm launch
    demo.launch(
        share=True,
        server_name="0.0.0.0",  # Cho phép truy cập từ mọi địa chỉ IP
        server_port=7861        # Cổng mặc định của Gradio
    )

# Điểm vào chính của chương trình
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LatentSync Video Lipsync and Enhancement")
    parser.add_argument("--unet_config_path", type=str, default="configs/unet/second_stage.yaml", help="Path to UNet configuration file")
    parser.add_argument("--inference_ckpt_path", type=str, help="Path to the inference checkpoint")
    parser.add_argument("--video_path", type=str, help="Path to the input video")
    parser.add_argument("--audio_path", type=str, help="Path to the input audio")
    parser.add_argument("--video_out_path", type=str, help="Path to the output video")
    parser.add_argument("--inference_steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale for the model")
    parser.add_argument("--seed", type=int, default=1247, help="Random seed for reproducibility")
    parser.add_argument("--gradio", action="store_true", help="Run the Gradio interface")
    args = parser.parse_args()

    # Determine config path
    if args.gradio:
        config_path = os.getenv("DEFAULT_CONFIG_PATH", os.path.join(script_dir, "configs/unet/second_stage.yaml"))
    else:
        config_path = os.path.join(script_dir, args.unet_config_path)

    # Check if config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load configuration
    config = OmegaConf.load(config_path)

    if args.gradio:
        gradio_interface()
    else:
        # Check if required arguments are provided
        if not all([args.inference_ckpt_path, args.video_path, args.audio_path, args.video_out_path]):
            parser.error("When not using --gradio, the following arguments are required: --inference_ckpt_path, --video_path, --audio_path, --video_out_path")
        main(config, args)