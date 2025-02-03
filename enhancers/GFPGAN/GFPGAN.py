import cv2
import onnxruntime
import numpy as np

class GFPGAN:
    def __init__(self, model_path="GFPGANv1.4.onnx", device='cpu'):
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        if device == 'cuda':
            providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}), "CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=providers)
        self.resolution = self.session.get_inputs()[0].shape[-2:]

    def preprocess(self, imgs):
        # imgs: list or numpy array of images (N, H, W, C)
        imgs_resized = np.array([cv2.resize(img, self.resolution, interpolation=cv2.INTER_LINEAR) for img in imgs])
        imgs_normalized = imgs_resized.astype(np.float32)[:, :, :, ::-1] / 255.0  # Convert BGR to RGB and normalize
        imgs_normalized = imgs_normalized.transpose(0, 3, 1, 2)  # Convert to NCHW
        imgs_normalized = (imgs_normalized - 0.5) / 0.5  # Normalize to [-1, 1]
        return imgs_normalized.astype(np.float32)

    def postprocess(self, imgs):
        # imgs: numpy array of shape (N, C, H, W)
        imgs = imgs.transpose(0, 2, 3, 1)  # Convert to NHWC
        imgs = (imgs.clip(-1, 1) + 1) * 0.5  # Scale to [0, 1]
        imgs = (imgs * 255)[:, :, :, ::-1]  # Convert RGB to BGR
        imgs = imgs.clip(0, 255).astype('uint8')
        return imgs

    def enhance_batch(self, imgs):
        # imgs: list or numpy array of images (N, H, W, C)
        imgs_preprocessed = self.preprocess(imgs)
        outputs = self.session.run(None, {'input': imgs_preprocessed})[0]
        imgs_postprocessed = self.postprocess(outputs)
        return imgs_postprocessed

    def enhance(self, img):
        # Process a single image
        return self.enhance_batch([img])[0]