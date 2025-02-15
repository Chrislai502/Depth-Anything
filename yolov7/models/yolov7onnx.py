import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from pathlib import Path
import numpy as np
import cv2

# Import YOLOv7 model loading function. This assumes you have the YOLOv7 repository 
# (or at least the 'models' folder) in your PYTHONPATH.
from models.experimental import attempt_load


class Yolov7Onnx(nn.Module):
    def __init__(
        self,
        pt_weights,            # Path to the .pt weights file
        onnx_path = "/home/chris/testing/Depth-Anything/yolov7/transferred_checkpts/yolov7_p5_transfer_flc8/best_ptq_folded.onnx/home/chris/testing/Depth-Anything/yolov7/transferred_checkpts/yolov7_p5_transfer_flc8/best_ptq_folded.onnx/home/chris/testing/Depth-Anything/yolov7/transferred_checkpts/yolov7_p5_transfer_flc8/best_ptq_folded.onnx/home/chris/testing/Depth-Anything/yolov7/transferred_checkpts/yolov7_p5_transfer_flc8/best_ptq_folded.onnx",             # Path where the ONNX file will be saved
        img_size=(640, 640),   # Inference image size (height, width)
        batch_size=1,
        device='cpu',          # 'cpu' or 'cuda'
        dynamic_batch=True,    # Whether to export with dynamic batch size
        grid=False,            # If using grid export (affects output naming)
        end2end=False,         # For end-to-end ONNX export (if needed)
        simplify=False,        # If True, attempt to simplify the ONNX model
        **export_kwargs        # Any extra kwargs for exporting (if desired)
    ):
        super(Yolov7Onnx, self).__init__()
        self.pt_weights = pt_weights
        self.onnx_path = onnx_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.device = device
        self.dynamic_batch = dynamic_batch
        self.grid = grid
        self.end2end = end2end
        self.simplify = simplify

        # Load the YOLOv7 PyTorch model
        self.model = attempt_load(self.pt_weights, map_location=device)
        self.model.eval()

        # If the ONNX file does not exist, export it now
        if not Path(self.onnx_path).exists():
            self.export_to_onnx(**export_kwargs)

        # Create an ONNX Runtime session for inference
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device != 'cpu' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(self.onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def export_to_onnx(self, **kwargs):
        """
        Exports the loaded PyTorch model to an ONNX file.
        Uses a dummy input of shape [batch_size, 3, H, W] and allows dynamic batch axes if desired.
        Optionally simplifies the ONNX model.
        """
        dummy_input = torch.zeros(self.batch_size, 3, self.img_size[0], self.img_size[1]).to(self.device)

        # Set dynamic axes if desired
        dynamic_axes = None
        if self.dynamic_batch:
            dynamic_axes = {'images': {0: 'batch'}}
            # If not using the grid export mode, assume one output tensor named "output"
            if not self.grid:
                dynamic_axes.update({'output': {0: 'batch'}})

        # Define output names (you may adjust this if your export requires different naming)
        output_names = ['output']

        # Export the model. (For more advanced exports, see export.py in the YOLOv7 repo.)
        torch.onnx.export(
            self.model,
            dummy_input,
            self.onnx_path,
            verbose=False,
            opset_version=12,
            input_names=['images'],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            **kwargs
        )

        # Optionally simplify the ONNX model
        if self.simplify:
            try:
                import onnxsim
                onnx_model = onnx.load(self.onnx_path)
                onnx_model, check = onnxsim.simplify(onnx_model)
                if not check:
                    raise RuntimeError("Simplified ONNX model could not be validated")
                onnx.save(onnx_model, self.onnx_path)
            except Exception as e:
                print("ONNX simplification failed:", e)

    def forward(self, x):
        """
        Runs inference using the ONNX Runtime session.
        Expects x to be a numpy array or torch.Tensor of shape [batch, 3, H, W].
        If the tensor values are not in the [0,1] range, they will be normalized.
        """
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        # Normalize if necessary (assumes input should be 0-1)
        if x.max() > 1.0:
            x = x / 255.0
        outputs = self.session.run(self.output_names, {self.input_name: x})
        return outputs

    @staticmethod
    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        """
        Resizes and pads an image while meeting stride-multiple constraints.
        This function mimics the preprocessing used in YOLOv7.
        """
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        # Compute new unpadded shape and padding
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, r, (dw, dh)

    def preprocess(self, img):
        """
        Preprocesses an input image (in BGR format as read by cv2) by:
         - Converting to RGB,
         - Resizing and padding with letterbox,
         - Converting from HWC to CHW format,
         - Adding a batch dimension,
         - Converting to float32.
        Returns the preprocessed image along with the scaling ratio and padding info.
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, ratio, dwdh = self.letterbox(img, new_shape=self.img_size, auto=False)
        img = img.transpose((2, 0, 1))  # convert HWC to CHW
        img = np.expand_dims(img, 0)
        img = np.ascontiguousarray(img, dtype=np.float32)
        return img, ratio, dwdh


# === Example usage ===
if __name__ == '__main__':
    # Paths for the weights and the ONNX model.
    pt_weights = './yolov7-tiny.pt'
    onnx_path = './yolov7-tiny.onnx'
    
    teacher = Yolov7Onnx(
        pt_weights,
        onnx_path,
        img_size=(640, 640),
        batch_size=1,
        device='cuda',         # Use 'cpu' if GPU is not available
        dynamic_batch=True,
        grid=True,
        end2end=True,
        simplify=True,
        topk_all=100,
        iou_thres=0.65,
        conf_thres=0.35,
        max_wh=7680
    )
    
    # Read an image and preprocess it
    img_path = 'inference/images/horses.jpg'
    img = cv2.imread(img_path)
    img_pre, ratio, dwdh = teacher.preprocess(img)
    
    # Run inference with the ONNX runtime wrapped in the teacher model
    outputs = teacher.forward(img_pre)
    print("Model outputs:", outputs)
