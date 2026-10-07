import os, glob, sys
import logging

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import normalize
from torchvision.ops import masks_to_boxes

import numpy as np
import cv2
import math
from typing import List
from PIL import Image
import io
from scipy import stats
from insightface.app.common import Face

from modules.processing import ProcessingImg2Img
from modules.shared import state
# from comfy_extras.chainner_models import model_loading
import comfy.model_management as model_management
import comfy.utils
import folder_paths

import scripts.reactor_version
from r_chainner import model_loading
from scripts.reactor_swapper import insightface_path
from scripts.reactor_faceswap import (
    FaceSwapScript,
    get_models,
    get_current_faces_model,
    analyze_faces,
    half_det_size,
    providers
)
from scripts.reactor_swapper import (
    unload_all_models,
    smooth_blend_values,
    smooth_face_angles,
)
from scripts.reactor_logger import logger
from reactor_utils import (
    batch_tensor_to_pil,
    batched_pil_to_tensor,
    tensor_to_pil,
    img2tensor,
    tensor2img,
    save_face_model,
    load_face_model,
    download,
    set_ort_session,
    prepare_cropped_face,
    normalize_cropped_face,
    add_folder_path_and_extensions,
    rgba2rgb_tensor,
    progress_bar,
    progress_bar_reset
)
from reactor_patcher import apply_patch
from r_facelib.utils.face_restoration_helper import FaceRestoreHelper
from r_basicsr.utils.registry import ARCH_REGISTRY
import scripts.r_archs.codeformer_arch
import scripts.r_masking.core as core

# NSFW check disabled: skip importing reactor_sfw (avoids loading transformers at startup)
# import scripts.reactor_sfw as sfw


models_dir = folder_paths.models_dir
REACTOR_MODELS_PATH = os.path.join(models_dir, "reactor")
FACE_MODELS_PATH = os.path.join(REACTOR_MODELS_PATH, "faces")
NSFWDET_MODEL_PATH = os.path.join(models_dir, "nsfw_detector","vit-base-nsfw-detector")

if not os.path.exists(REACTOR_MODELS_PATH):
    os.makedirs(REACTOR_MODELS_PATH)
    if not os.path.exists(FACE_MODELS_PATH):
        os.makedirs(FACE_MODELS_PATH)

dir_facerestore_models = os.path.join(models_dir, "facerestore_models")
os.makedirs(dir_facerestore_models, exist_ok=True)
folder_paths.folder_names_and_paths["facerestore_models"] = ([dir_facerestore_models], folder_paths.supported_pt_extensions)

BLENDED_FACE_MODEL = None
FACE_SIZE: int = 512
FACE_HELPER = None

if "ultralytics" not in folder_paths.folder_names_and_paths:
    add_folder_path_and_extensions("ultralytics_bbox", [os.path.join(models_dir, "ultralytics", "bbox")], folder_paths.supported_pt_extensions)
    add_folder_path_and_extensions("ultralytics_segm", [os.path.join(models_dir, "ultralytics", "segm")], folder_paths.supported_pt_extensions)
    add_folder_path_and_extensions("ultralytics", [os.path.join(models_dir, "ultralytics")], folder_paths.supported_pt_extensions)
if "sams" not in folder_paths.folder_names_and_paths:
    add_folder_path_and_extensions("sams", [os.path.join(models_dir, "sams")], folder_paths.supported_pt_extensions)

def get_facemodels():
    models_path = os.path.join(FACE_MODELS_PATH, "*")
    models = glob.glob(models_path)
    models = [x for x in models if x.endswith(".safetensors")]
    return models

def get_restorers():
    models_path = os.path.join(models_dir, "facerestore_models/*")
    models = glob.glob(models_path)
    models = [x for x in models if (x.endswith(".pth") or x.endswith(".onnx"))]
    if len(models) == 0:
        fr_urls = [
            # "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.3.pth",
            "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.4.pth",
            # "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/codeformer-v0.1.0.pth",
            # "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-512.onnx",
        ]
        for model_url in fr_urls:
            model_name = os.path.basename(model_url)
            model_path = os.path.join(dir_facerestore_models, model_name)
            download(model_url, model_path, model_name)
        models = glob.glob(models_path)
        models = [x for x in models if (x.endswith(".pth") or x.endswith(".onnx"))]
    return models

def get_model_names(get_models):
    models = get_models()
    names = []
    for x in models:
        names.append(os.path.basename(x))
    names.sort(key=str.lower)
    names.insert(0, "none")
    return names

def model_names():
    models = get_models()
    return {os.path.basename(x): x for x in models}


def build_swap_mask(input_image, result, eps=0.05, dilate_px=6):
    """换脸区域 = |swapped - original| 的 RGB 均值 > eps，再膨胀 dilate_px。
    换脸（含 GFPGAN/blend）只在脸部区域改像素，两图在区域外逐像素相同——
    差分就是换脸区域的像素级精确描述，零检测、零模型。"""
    k = dilate_px * 2 + 1
    masks = []
    step = 8
    for s in range(0, int(result.shape[0]), step):
        e = min(s + step, int(result.shape[0]))
        # result 在计算设备上（换脸尾部），input 通常在 CPU——统一到 result 的设备再差分
        r = result[s:e].float()
        i = input_image[s:e].float().to(r.device)
        d = (r - i).abs().mean(dim=-1)
        m = (d > eps).float().unsqueeze(1)
        m = F.max_pool2d(m, kernel_size=k, stride=1, padding=dilate_px)
        masks.append(m.squeeze(1).cpu())
    return torch.cat(masks, dim=0) if masks else None


class reactor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True, "label_off": "OFF", "label_on": "ON"}),
                "input_image": ("IMAGE",),
                "swap_model": (list(model_names().keys()),),
                "facedetection": (["retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n"],),
                "face_restore_model": (get_model_names(get_restorers),),
                "face_restore_visibility": ("FLOAT", {"default": 1, "min": 0.1, "max": 1, "step": 0.05}),
                "codeformer_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.05}),
                "detect_gender_input": (["no","female","male"], {"default": "no"}),
                "detect_gender_source": (["no","female","male"], {"default": "no"}),
                "input_faces_index": ("STRING", {"default": "0"}),
                "source_faces_index": ("STRING", {"default": "0"}),
                "console_log_level": ([0, 1, 2], {"default": 1}),
            },
            "optional": {
                "source_image": ("IMAGE",),
                "face_model": ("FACE_MODEL",),
                "face_boost": ("FACE_BOOST",),
            },
            "hidden": {"faces_order": "FACES_ORDER"},
        }

    RETURN_TYPES = ("IMAGE","FACE_MODEL","IMAGE","MASK")
    RETURN_NAMES = ("SWAPPED_IMAGE","FACE_MODEL","ORIGINAL_IMAGE","SWAP_MASK")
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor"

    def __init__(self):
        # self.face_helper = None
        self.faces_order = ["large-small", "large-small"]
        # self.face_size = FACE_SIZE
        self.face_boost_enabled = False
        self.restore = True
        self.boost_model = None
        self.interpolation = "Bicubic"
        self.boost_model_visibility = 1
        self.boost_cf_weight = 0.5
        self.last_swapped_bboxes = None
        # self.last_swapped_indices = None
        self.restore_swapped_only = True

    def restore_face(
        self,
        input_image,
        face_restore_model,
        face_restore_visibility,
        codeformer_weight,
        facedetection,
    ):

        # from datetime import datetime
        # def _current_time():
        #     current_datetime = datetime.now()
        #     return current_datetime.strftime("%M:%S")
        
        # локальная функция IoU
        def _iou(b1, b2):
            xA = max(b1[0], b2[0])
            yA = max(b1[1], b2[1])
            xB = min(b1[2], b2[2])
            yB = min(b1[3], b2[3])
            interW = max(0.0, xB - xA)
            interH = max(0.0, yB - yA)
            interArea = interW * interH
            if interArea == 0:
                return 0.0
            boxAArea = max(1.0, (b1[2]-b1[0]) * (b1[3]-b1[1]))
            boxBArea = max(1.0, (b2[2]-b2[0]) * (b2[3]-b2[1]))
            return interArea / float(boxAArea + boxBArea - interArea)
        
        result = input_image

        if face_restore_model != "none" and not model_management.processing_interrupted():

            global FACE_SIZE, FACE_HELPER

            self.face_helper = FACE_HELPER

            faceSize = 512
            if "1024" in face_restore_model.lower():
                faceSize = 1024
            elif "2048" in face_restore_model.lower():
                faceSize = 2048

            logger.status(f"Restoring with {face_restore_model} | Face Size is set to {faceSize}")

            model_path = folder_paths.get_full_path("facerestore_models", face_restore_model)

            device = model_management.get_torch_device()

            if "codeformer" in face_restore_model.lower():

                codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
                    dim_embd=512,
                    codebook_size=1024,
                    n_head=8,
                    n_layers=9,
                    connect_list=["32", "64", "128", "256"],
                ).to(device)
                checkpoint = torch.load(model_path)["params_ema"]
                codeformer_net.load_state_dict(checkpoint)
                facerestore_model = codeformer_net.eval()

            elif ".onnx" in face_restore_model:

                ort_session = set_ort_session(model_path, providers=providers)
                ort_session_inputs = {}
                facerestore_model = ort_session

            else:

                sd = comfy.utils.load_torch_file(model_path, safe_load=True)
                facerestore_model = model_loading.load_state_dict(sd).eval()
                facerestore_model.to(device)

            if faceSize != FACE_SIZE or self.face_helper is None:
                self.face_helper = FaceRestoreHelper(1, face_size=faceSize, crop_ratio=(1, 1), det_model=facedetection, save_ext='png', use_parse=True, device=device)
                FACE_SIZE = faceSize
                FACE_HELPER = self.face_helper

            # Copying Tensor to CPU (if it isn't) to convert torch.Tensor to np.ndarray
            image_np = 255. * result.cpu().numpy()

            total_images = image_np.shape[0]

            out_images = []
            
            pbar = progress_bar(total_images)

            for i in range(total_images):

                # if total_images > 1:
                #     logger.status(f"Restoring {i}")

                cur_image_np = image_np[i,:, :, ::-1]

                original_resolution = cur_image_np.shape[0:2]

                if facerestore_model is None or self.face_helper is None:
                    return result

                self.face_helper.clean_all()
                self.face_helper.read_image(cur_image_np)
                self.face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
                self.face_helper.align_warp_face()
                
                # restored_face = None
                restored_faces = []
                
                # берем сохранённые bbox из swap (или None)
                # swap 阶段现按帧记录 bbox（每帧一个列表），此处只与当前帧
                # 实际换过脸的 bbox 匹配；未换脸的帧不再被相邻帧的 bbox
                # 误匹配而做 GFPGAN 修复（修复纹理突变也是闪烁来源之一）
                swapped_all = getattr(self, "last_swapped_bboxes", None)
                swapped_bboxes = None
                if swapped_all:
                    if len(swapped_all) == total_images and isinstance(swapped_all[0], (list, tuple)):
                        frame_boxes = swapped_all[i] if i < len(swapped_all) else []
                        if frame_boxes and isinstance(frame_boxes[0], (int, float)):
                            # 兼容旧版扁平结构（元素是坐标而非 bbox 列表）
                            swapped_bboxes = swapped_all
                        else:
                            swapped_bboxes = frame_boxes
                    else:
                        # 兼容旧版扁平结构
                        swapped_bboxes = swapped_all
                # флаги, чтобы одно сохранённое bbox не совпало с несколькими лицами
                used_swapped = [False] * len(swapped_bboxes) if swapped_bboxes else None

                IOU_THRESHOLD = 0.5

                for idx, cropped_face in enumerate(self.face_helper.cropped_faces):

                    # определяем bbox текущего лица, который дал детектор внутри FaceRestoreHelper
                    current_bbox = None
                    if hasattr(self.face_helper, 'det_faces') and len(self.face_helper.det_faces) > idx:
                        det = self.face_helper.det_faces[idx]
                        # det м.б. [x1,y1,x2,y2,score]
                        current_bbox = (float(det[0]), float(det[1]), float(det[2]), float(det[3]))

                    # логика: если у нас есть сохранённые bbox — ресторим только если iou с одним из них > порог
                    do_restore = True
                    # if self.last_swapped_indices is not None:
                    #     do_restore = idx in self.last_swapped_indices
                    if swapped_bboxes and self.restore_swapped_only:
                        do_restore = False
                        if current_bbox is not None:
                            for s_idx, sbox in enumerate(swapped_bboxes):
                                if used_swapped is not None and used_swapped[s_idx]:
                                    continue
                                if _iou(current_bbox, sbox) >= IOU_THRESHOLD:
                                    cx1 = (current_bbox[0] + current_bbox[2]) / 2
                                    cy1 = (current_bbox[1] + current_bbox[3]) / 2
                                    cx2 = (sbox[0] + sbox[2]) / 2
                                    cy2 = (sbox[1] + sbox[3]) / 2
                                    if abs(cx1 - cx2) < (current_bbox[2] - current_bbox[0]) * 0.25:
                                        if abs(cy1 - cy2) < (current_bbox[3] - current_bbox[1]) * 0.25:
                                            do_restore = True
                                    # do_restore = True
                                    if used_swapped is not None:
                                        used_swapped[s_idx] = True
                                    break
                    
                    if do_restore:
                    
                        # if ".pth" in face_restore_model:
                        cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
                        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

                        try:

                            with torch.no_grad():

                                if ".onnx" in face_restore_model: # ONNX models

                                    for ort_session_input in ort_session.get_inputs():
                                        if ort_session_input.name == "input":
                                            cropped_face_prep = prepare_cropped_face(cropped_face)
                                            ort_session_inputs[ort_session_input.name] = cropped_face_prep
                                        if ort_session_input.name == "weight":
                                            weight = np.array([ 1 ], dtype = np.double)
                                            ort_session_inputs[ort_session_input.name] = weight

                                    output = ort_session.run(None, ort_session_inputs)[0][0]
                                    restored_face = normalize_cropped_face(output)

                                else: # PTH models

                                    output = facerestore_model(cropped_face_t, w=codeformer_weight)[0] if "codeformer" in face_restore_model.lower() else facerestore_model(cropped_face_t)[0]
                                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))

                            del output
                            torch.cuda.empty_cache()

                        except Exception as error:

                            print(f"\tFailed inference: {error}", file=sys.stderr)
                            # restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))
                            restored_face = cropped_face.copy()
                        
                    else:
                        restored_face = cropped_face.copy()
                    
                    if face_restore_visibility < 1:
                        restored_face = cropped_face * (1 - face_restore_visibility) + restored_face * face_restore_visibility

                    restored_face = restored_face.astype("uint8")
                    self.face_helper.add_restored_face(restored_face)
                
                self.face_helper.get_inverse_affine(None)

                restored_img = self.face_helper.paste_faces_to_input_image()
                restored_img = restored_img[:, :, ::-1]

                if original_resolution != restored_img.shape[0:2]:
                    restored_img = cv2.resize(restored_img, (0, 0), fx=original_resolution[1]/restored_img.shape[1], fy=original_resolution[0]/restored_img.shape[0], interpolation=cv2.INTER_AREA)

                self.face_helper.clean_all()

                # out_images[i] = restored_img
                out_images.append(restored_img)

                if state.interrupted or model_management.processing_interrupted():
                    logger.status("Interrupted by User")
                    return input_image
                
                pbar.update(1)

            restored_img_np = np.array(out_images).astype(np.float32) / 255.0
            restored_img_tensor = torch.from_numpy(restored_img_np)

            result = restored_img_tensor

            progress_bar_reset(pbar)

        # if hasattr(self, "last_swapped_bboxes"):
        self.last_swapped_bboxes = None
        # if hasattr(self, "last_swapped_indices"):
        # self.last_swapped_indices = None
        
        return result


    def execute(self, enabled, input_image, swap_model, detect_gender_source, detect_gender_input, source_faces_index, input_faces_index, console_log_level, face_restore_model,face_restore_visibility, codeformer_weight, facedetection, source_image=None, face_model=None, faces_order=None, face_boost=None, angle_threshold=60.0):

        device = model_management.get_torch_device()

        if isinstance(input_image, torch.Tensor) and input_image.device != device:
            input_image = input_image.to(device)

        if face_boost is not None:
            self.face_boost_enabled = face_boost["enabled"]
            self.boost_model = face_boost["boost_model"]
            self.interpolation = face_boost["interpolation"]
            self.boost_model_visibility = face_boost["visibility"]
            self.boost_cf_weight = face_boost["codeformer_weight"]
            self.restore = face_boost["restore_with_main_after"]
        else:
            self.face_boost_enabled = False

        if faces_order is None:
            faces_order = self.faces_order

        apply_patch(console_log_level)

        if not enabled:
            return (input_image,face_model)
        elif source_image is None and face_model is None:
            logger.error("Please provide 'source_image' or `face_model`")
            return (input_image,face_model)

        if face_model == "none":
            face_model = None

        script = FaceSwapScript()
        pil_images = batch_tensor_to_pil(input_image)

        # NSFW checker (disabled: no model load, no PNG re-encoding per image)
        # logger.status("Checking for any unsafe content...")
        # pbar = progress_bar(len(pil_images))
        # pil_images_sfw = []
        # for img in pil_images:
        #     if state.interrupted or model_management.processing_interrupted():
        #         logger.status("Interrupted by User")
        #         break
        #     img_byte_arr = io.BytesIO()
        #     img.save(img_byte_arr, format='PNG')
        #     img_byte_arr = img_byte_arr.getvalue()
        #     if not sfw.nsfw_image(img_byte_arr, NSFWDET_MODEL_PATH):
        #         pil_images_sfw.append(img)
        #     pbar.update(1)
        # pil_images = pil_images_sfw
        # # #
        # progress_bar_reset(pbar)

        if len(pil_images) > 0:

            if source_image is not None:
                source = tensor_to_pil(source_image)
            else:
                source = None
            
            p = ProcessingImg2Img(pil_images)
            script.process(
                p=p,
                img=source,
                enable=True,
                source_faces_index=source_faces_index,
                faces_index=input_faces_index,
                model=swap_model,
                swap_in_source=True,
                swap_in_generated=True,
                gender_source=detect_gender_source,
                gender_target=detect_gender_input,
                face_model=face_model,
                faces_order=faces_order,
                # face boost:
                face_boost_enabled=self.face_boost_enabled,
                face_restore_model=self.boost_model,
                face_restore_visibility=self.boost_model_visibility,
                codeformer_weight=self.boost_cf_weight,
                interpolation=self.interpolation,
                angle_threshold=angle_threshold,
            )
            result = batched_pil_to_tensor(p.init_images)
            # print(f"bbox={p.bbox}")
            if len(p.bbox) > 0:
                # 统一为"每帧一组bbox"的结构，供 restore_face 按帧精确匹配；
                # 扁平结构（单图路径）包装成单帧
                first = p.bbox[0]
                if isinstance(first, (tuple, list)) and len(first) == 4 and all(isinstance(v, (int, float)) for v in first):
                    self.last_swapped_bboxes = [p.bbox]
                else:
                    self.last_swapped_bboxes = p.bbox
                # self.last_swapped_indices = p.swapped_indexes
            else:
                self.last_swapped_bboxes = None
            original_image = input_image

            if face_model is None:
                current_face_model = get_current_faces_model()
                face_model_to_provide = current_face_model[0] if (current_face_model is not None and len(current_face_model) > 0) else face_model
            else:
                face_model_to_provide = face_model

            if self.restore or not self.face_boost_enabled:
                result = reactor.restore_face(self,result,face_restore_model,face_restore_visibility,codeformer_weight,facedetection)

            # 应用平滑blend
            if hasattr(p, 'face_angles') and len(p.face_angles) > 0 and len(original_image) == len(p.face_angles):
                logger.status("Applying smooth blend based on face angles...")
                # 对角度序列进行平滑处理，减少检测抖动
                smoothed_angles = p.face_angles
                # 优先使用 swap 阶段预计算的 blend 权重（与软跳过逻辑完全一致）
                pre_weights = getattr(p, 'face_blend_weights', None)
                if pre_weights is not None and len(pre_weights) == len(smoothed_angles):
                    smoothed_weights = pre_weights
                else:
                    smoothed_weights = smooth_blend_values(smoothed_angles, angle_threshold)
                
                # 在float32格式下进行blend操作，避免精度损失
                result_np = result.cpu().numpy().astype(np.float32)
                original_np = original_image.cpu().numpy().astype(np.float32)
                
                blended_results = []
                for i, (result_img, orig_img, weight) in enumerate(zip(result_np, original_np, smoothed_weights)):
                    if weight > 0.01:
                        blended = orig_img * weight + result_img * (1.0 - weight)
                        blended_results.append(blended)
                        # logger.info(f"Frame {i}: angle={p.face_angles[i]:.1f}°, original_weight={weight:.3f}")
                    else:
                        blended_results.append(result_img)
                
                # 打印各帧的角度和blend系数
                angle_weight_str = ",".join([f"{a:.1f}-{w:.1f}" for a, w in zip(smoothed_angles, smoothed_weights)])
                logger.status(f"Angles & weights: {angle_weight_str}")
                
                # 转换回tensor
                blended_np = np.array(blended_results).astype(np.float32)
                result = torch.from_numpy(blended_np).to(result.device)

        else:
            image_black = Image.new("RGB", (512, 512))
            result = batched_pil_to_tensor([image_black])
            face_model_to_provide = None
            original_image = result

        # 换脸区域 mask：|result - original| 的像素级差分（含 GFPGAN/blend 的改动）
        swap_mask = build_swap_mask(original_image, result)
        return (result,face_model_to_provide,original_image,swap_mask)


class ReActorPlusOpt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True, "label_off": "OFF", "label_on": "ON"}),
                "input_image": ("IMAGE",),
                "swap_model": (list(model_names().keys()),),
                "facedetection": (["retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n"],),
                "face_restore_model": (get_model_names(get_restorers),),
                "face_restore_visibility": ("FLOAT", {"default": 1, "min": 0.1, "max": 1, "step": 0.05}),
                "codeformer_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.05}),
                "angle_threshold": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 90.0, "step": 1.0}),
            },
            "optional": {
                "source_image": ("IMAGE",),
                "face_model": ("FACE_MODEL",),
                "options": ("OPTIONS",),
                "face_boost": ("FACE_BOOST",),
            }
        }

    RETURN_TYPES = ("IMAGE","FACE_MODEL","IMAGE","MASK")
    RETURN_NAMES = ("SWAPPED_IMAGE","FACE_MODEL","ORIGINAL_IMAGE","SWAP_MASK")
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor"

    def __init__(self):
        # self.face_helper = None
        self.faces_order = ["large-small", "large-small"]
        self.detect_gender_input = "no"
        self.detect_gender_source = "no"
        self.input_faces_index = "0"
        self.source_faces_index = "0"
        self.console_log_level = 1
        self.restore_swapped_only = True
        # self.face_size = 512
        self.face_boost_enabled = False
        self.restore = True
        self.boost_model = None
        self.interpolation = "Bicubic"
        self.boost_model_visibility = 1
        self.boost_cf_weight = 0.5

    def execute(self, enabled, input_image, swap_model, facedetection, face_restore_model, face_restore_visibility, codeformer_weight, source_image=None, face_model=None, options=None, face_boost=None, angle_threshold=60.0):

        if options is not None:
            self.faces_order = [options["input_faces_order"], options["source_faces_order"]]
            self.console_log_level = options["console_log_level"]
            self.detect_gender_input = options["detect_gender_input"]
            self.detect_gender_source = options["detect_gender_source"]
            self.input_faces_index = options["input_faces_index"]
            self.source_faces_index = options["source_faces_index"]
            self.restore_swapped_only = options["restore_swapped_only"]

        if face_boost is not None:
            self.face_boost_enabled = face_boost["enabled"]
            self.restore = face_boost["restore_with_main_after"]
        else:
            self.face_boost_enabled = False

        result = reactor.execute(
            self,enabled,input_image,swap_model,self.detect_gender_source,self.detect_gender_input,self.source_faces_index,self.input_faces_index,self.console_log_level,face_restore_model,face_restore_visibility,codeformer_weight,facedetection,source_image,face_model,self.faces_order, face_boost=face_boost, angle_threshold=angle_threshold
        )

        return result


class ReActorPlusOptWithDirection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True, "label_off": "OFF", "label_on": "ON"}),
                "input_image": ("IMAGE",),
                "swap_model": (list(model_names().keys()),),
                "facedetection": (["retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n"],),
                "face_restore_model": (get_model_names(get_restorers),),
                "face_restore_visibility": ("FLOAT", {"default": 1, "min": 0.1, "max": 1, "step": 0.05}),
                "codeformer_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.05}),
                "angle_threshold": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 90.0, "step": 1.0}),
            },
            "optional": {
                "source_image": ("IMAGE",),
                "face_model": ("FACE_MODEL",),
                "options": ("OPTIONS",),
                "face_boost": ("FACE_BOOST",),
            }
        }

    RETURN_TYPES = ("IMAGE","FACE_MODEL","IMAGE","MASK")
    RETURN_NAMES = ("SWAPPED_IMAGE","FACE_MODEL","ORIGINAL_IMAGE","SWAP_MASK")
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor"

    def __init__(self):
        self.faces_order = ["large-small", "large-small"]
        self.detect_gender_input = "no"
        self.detect_gender_source = "no"
        self.input_faces_index = "0"
        self.source_faces_index = "0"
        self.console_log_level = 1
        self.restore_swapped_only = True
        self.face_boost_enabled = False
        self.restore = True
        self.boost_model = None
        self.interpolation = "Bicubic"
        self.boost_model_visibility = 1
        self.boost_cf_weight = 0.5



    def execute(self, enabled, input_image, swap_model, facedetection, face_restore_model, face_restore_visibility, codeformer_weight, angle_threshold, source_image=None, face_model=None, options=None, face_boost=None):

        # 处理基本选项
        if options is not None:
            self.faces_order = [options["input_faces_order"], options["source_faces_order"]]
            self.console_log_level = options["console_log_level"]
            self.detect_gender_input = options["detect_gender_input"]
            self.detect_gender_source = options["detect_gender_source"]
            self.input_faces_index = options["input_faces_index"]
            self.source_faces_index = options["source_faces_index"]
            self.restore_swapped_only = options["restore_swapped_only"]

        # 处理人脸增强选项
        if face_boost is not None:
            self.face_boost_enabled = face_boost["enabled"]
            self.restore = face_boost["restore_with_main_after"]
        else:
            self.face_boost_enabled = False

        # 执行人脸替换
        return reactor.execute(
            self,enabled,input_image,swap_model,self.detect_gender_source,self.detect_gender_input,self.source_faces_index,self.input_faces_index,self.console_log_level,face_restore_model,face_restore_visibility,codeformer_weight,facedetection,source_image,face_model,self.faces_order, face_boost=face_boost, angle_threshold=angle_threshold
        )


class LoadFaceModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_model": (get_model_names(get_facemodels),),
            }
        }

    RETURN_TYPES = ("FACE_MODEL","STRING")
    RETURN_NAMES = ("FACE_MODEL","FACE_MODEL_NAME")
    FUNCTION = "load_model"
    CATEGORY = "🌌 ReActor"

    def load_model(self, face_model):
        self.face_model = face_model
        face_model = face_model.split(".safetensors")[0] if ".safetensors" in face_model else face_model
        self.face_models_path = FACE_MODELS_PATH
        if self.face_model != "none":
            face_model_path = os.path.join(self.face_models_path, self.face_model)
            out = load_face_model(face_model_path)
        else:
            out = None
        return (out,face_model)


class ReActorWeight:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "faceswap_weight": (["0%", "12.5%", "25%", "37.5%", "50%", "62.5%", "75%", "87.5%", "100%"], {"default": "50%"}),
            },
            "optional": {
                "source_image": ("IMAGE",),
                "face_model": ("FACE_MODEL",),
            }
        }
    
    RETURN_TYPES = ("IMAGE","FACE_MODEL")
    RETURN_NAMES = ("INPUT_IMAGE","FACE_MODEL")
    FUNCTION = "set_weight"

    OUTPUT_NODE = True

    CATEGORY = "🌌 ReActor"

    def set_weight(self, input_image, faceswap_weight, face_model=None, source_image=None):

        if input_image is None:
            logger.error("Please provide `input_image`")
            return (input_image,None)
        
        if source_image is None and face_model is None:
            logger.error("Please provide `source_image` or `face_model`")
            return (input_image,None)

        weight = float(faceswap_weight.split("%")[0])

        images = []
        faces = [] if face_model is None else [face_model]
        embeddings = [] if face_model is None else [face_model.embedding]

        if weight == 0:
            images = [input_image]
            faces = []
            embeddings = []
        elif weight == 100:
            if face_model is None:
                images = [source_image]
        else:
            if weight > 50:
                images = [input_image]
                count = round(100/(100-weight))
            else:
                if face_model is None:
                    images = [source_image]
                count = round(100/(weight))
            for i in range(count-1):
                if weight > 50:
                    if face_model is None:
                        images.append(source_image)
                    else:
                        faces.append(face_model)
                        embeddings.append(face_model.embedding)
                else:
                    images.append(input_image)
        
        images_list: List[Image.Image] = []

        apply_patch(1)

        if len(images) > 0:

            for image in images:
                img = tensor_to_pil(image)
                images_list.append(img)

            for image in images_list:
                face = BuildFaceModel.build_face_model(self,image)
                if isinstance(face, str):
                    continue
                faces.append(face)
                embeddings.append(face.embedding)
        
        if len(faces) > 0:
            blended_embedding = np.mean(embeddings, axis=0)
            blended_face = Face(
                bbox=faces[0].bbox,
                kps=faces[0].kps,
                det_score=faces[0].det_score,
                landmark_3d_68=faces[0].landmark_3d_68,
                pose=faces[0].pose,
                landmark_2d_106=faces[0].landmark_2d_106,
                embedding=blended_embedding,
                gender=faces[0].gender,
                age=faces[0].age
            )
            if blended_face is None:
                no_face_msg = "Something went wrong, please try another set of images"
                logger.error(no_face_msg)

        return (input_image,blended_face)


class BuildFaceModel:
    def __init__(self):
        self.output_dir = FACE_MODELS_PATH

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "save_mode": ("BOOLEAN", {"default": True, "label_off": "OFF", "label_on": "ON"}),
                "send_only": ("BOOLEAN", {"default": False, "label_off": "NO", "label_on": "YES"}),
                "face_model_name": ("STRING", {"default": "default"}),
                "compute_method": (["Mean", "Median", "Mode"], {"default": "Mean"}),
            },
            "optional": {
                "images": ("IMAGE",),
                "face_models": ("FACE_MODEL",),
            }
        }

    RETURN_TYPES = ("FACE_MODEL",)
    FUNCTION = "blend_faces"

    OUTPUT_NODE = True

    CATEGORY = "🌌 ReActor"

    def build_face_model(self, image: Image.Image, det_size=(640, 640)):
        logging.StreamHandler.terminator = "\n"
        if image is None:
            error_msg = "Please load an Image"
            logger.error(error_msg)
            return error_msg
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        face_model = analyze_faces(image, det_size)

        if len(face_model) == 0:
            print("")
            det_size_half = half_det_size(det_size)
            face_model = analyze_faces(image, det_size_half)
            if face_model is not None and len(face_model) > 0:
                print("...........................................................", end=" ")

        if face_model is not None and len(face_model) > 0:
            return face_model[0]
        else:
            no_face_msg = "No face found, please try another image"
            # logger.error(no_face_msg)
            return no_face_msg

    def blend_faces(self, save_mode, send_only, face_model_name, compute_method, images=None, face_models=None):
        global BLENDED_FACE_MODEL
        blended_face: Face = BLENDED_FACE_MODEL

        if send_only and blended_face is None:
            send_only = False

        if (images is not None or face_models is not None) and not send_only:

            faces = []
            embeddings = []

            apply_patch(1)

            if images is not None:
                images_list: List[Image.Image] = batch_tensor_to_pil(images)

                n = len(images_list)

                for i,image in enumerate(images_list):
                    logging.StreamHandler.terminator = " "
                    logger.status(f"Building Face Model {i+1} of {n}...")
                    face = self.build_face_model(image)
                    if isinstance(face, str):
                        logger.error(f"No faces found in image {i+1}, skipping")
                        continue
                    else:
                        print(f"{int(((i+1)/n)*100)}%")
                    faces.append(face)
                    embeddings.append(face.embedding)

            elif face_models is not None:

                n = len(face_models)

                for i,face_model in enumerate(face_models):
                    logging.StreamHandler.terminator = " "
                    logger.status(f"Extracting Face Model {i+1} of {n}...")
                    face = face_model
                    if isinstance(face, str):
                        logger.error(f"No faces found for face_model {i+1}, skipping")
                        continue
                    else:
                        print(f"{int(((i+1)/n)*100)}%")
                    faces.append(face)
                    embeddings.append(face.embedding)

            logging.StreamHandler.terminator = "\n"
            if len(faces) > 0:
                # compute_method_name = "Mean" if compute_method == 0 else "Median" if compute_method == 1 else "Mode"
                logger.status(f"Blending with Compute Method '{compute_method}'...")
                blended_embedding = np.mean(embeddings, axis=0) if compute_method == "Mean" else np.median(embeddings, axis=0) if compute_method == "Median" else stats.mode(embeddings, axis=0)[0].astype(np.float32)
                blended_face = Face(
                    bbox=faces[0].bbox,
                    kps=faces[0].kps,
                    det_score=faces[0].det_score,
                    landmark_3d_68=faces[0].landmark_3d_68,
                    pose=faces[0].pose,
                    landmark_2d_106=faces[0].landmark_2d_106,
                    embedding=blended_embedding,
                    gender=faces[0].gender,
                    age=faces[0].age
                )
                if blended_face is not None:
                    BLENDED_FACE_MODEL = blended_face
                    if save_mode:
                        face_model_path = os.path.join(FACE_MODELS_PATH, face_model_name + ".safetensors")
                        save_face_model(blended_face,face_model_path)
                        # done_msg = f"Face model has been saved to '{face_model_path}'"
                        # logger.status(done_msg)
                    logger.status("--Done!--")
                    # return (blended_face,)
                else:
                    no_face_msg = "Something went wrong, please try another set of images"
                    logger.error(no_face_msg)
                    # return (blended_face,)
            # logger.status("--Done!--")
        if images is None and face_models is None:
            logger.error("Please provide `images` or `face_models`")
        return (blended_face,)


class SaveFaceModel:
    def __init__(self):
        self.output_dir = FACE_MODELS_PATH

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "save_mode": ("BOOLEAN", {"default": True, "label_off": "OFF", "label_on": "ON"}),
                "face_model_name": ("STRING", {"default": "default"}),
                "select_face_index": ("INT", {"default": 0, "min": 0}),
            },
            "optional": {
                "image": ("IMAGE",),
                "face_model": ("FACE_MODEL",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_model"

    OUTPUT_NODE = True

    CATEGORY = "🌌 ReActor"

    def save_model(self, save_mode, face_model_name, select_face_index, image=None, face_model=None, det_size=(640, 640)):
        if save_mode and image is not None:
            source = tensor_to_pil(image)
            source = cv2.cvtColor(np.array(source), cv2.COLOR_RGB2BGR)
            apply_patch(1)
            logger.status("Building Face Model...")
            face_model_raw = analyze_faces(source, det_size)
            if len(face_model_raw) == 0:
                det_size_half = half_det_size(det_size)
                face_model_raw = analyze_faces(source, det_size_half)
            try:
                face_model = face_model_raw[select_face_index]
            except:
                logger.error("No face(s) found")
                return face_model_name
            logger.status("--Done!--")
        if save_mode and (face_model != "none" or face_model is not None):
            face_model_path = os.path.join(self.output_dir, face_model_name + ".safetensors")
            save_face_model(face_model,face_model_path)
        if image is None and face_model is None:
            logger.error("Please provide `face_model` or `image`")
        return face_model_name


class RestoreFace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "facedetection": (["retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n"],),
                "model": (get_model_names(get_restorers),),
                "visibility": ("FLOAT", {"default": 1, "min": 0.0, "max": 1, "step": 0.05}),
                "codeformer_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor"

    def execute(self, image, model, visibility, codeformer_weight, facedetection):
        result = reactor.restore_face(
            self, image, model, visibility, codeformer_weight, facedetection
        )
        return (result,)


class RestoreFaceAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "facedetection": (["retinaface_resnet50", "retinaface_mobile0.25", "YOLOv5l", "YOLOv5n"],),
                "model": (get_model_names(get_restorers),),
                "visibility": ("FLOAT", {"default": 1, "min": 0.0, "max": 1, "step": 0.05}),
                "codeformer_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.05}),
                "face_selection": (["all", "filter", "largest"],{"default": "all"}),
            },
            "optional": {
                "sort_by": (["area", "x_position", "y_position", "detection_confidence"],{"default": "area"}),
                "reverse_order": ("BOOLEAN", {"default": False}),
                "take_start": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "take_count": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor"

    def execute(
            self, image, model, visibility, codeformer_weight, facedetection, face_selection, sort_by="area", reverse_order=False, take_start=0, take_count=1
        ):

        min_x_position=0.0
        max_x_position=1.0
        min_y_position=0.0
        max_y_position=1.0

        result = image

        face_restore_model = model

        if face_restore_model != "none" and not model_management.processing_interrupted():

            global FACE_SIZE, FACE_HELPER

            self.face_helper = FACE_HELPER

            faceSize = 512
            if "1024" in face_restore_model.lower():
                faceSize = 1024
            elif "2048" in face_restore_model.lower():
                faceSize = 2048

            logger.status(f"Restoring with {face_restore_model} | Face Size is set to {faceSize}")

            model_path = folder_paths.get_full_path("facerestore_models", face_restore_model)

            device = model_management.get_torch_device()

            if "codeformer" in face_restore_model.lower():

                codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
                    dim_embd=512,
                    codebook_size=1024,
                    n_head=8,
                    n_layers=9,
                    connect_list=["32", "64", "128", "256"],
                ).to(device)
                checkpoint = torch.load(model_path)["params_ema"]
                codeformer_net.load_state_dict(checkpoint)
                facerestore_model = codeformer_net.eval()

            elif ".onnx" in face_restore_model:

                ort_session = set_ort_session(model_path, providers=providers)
                ort_session_inputs = {}
                facerestore_model = ort_session

            else:

                sd = comfy.utils.load_torch_file(model_path, safe_load=True)
                facerestore_model = model_loading.load_state_dict(sd).eval()
                facerestore_model.to(device)

            if faceSize != FACE_SIZE or self.face_helper is None:
                self.face_helper = FaceRestoreHelper(1, face_size=faceSize, crop_ratio=(1, 1), det_model=facedetection, save_ext='png', use_parse=True, device=device)
                FACE_SIZE = faceSize
                FACE_HELPER = self.face_helper

            image_np = 255. * result.cpu().numpy()

            total_images = image_np.shape[0]

            out_images = []
            
            pbar = progress_bar(total_images)

            for i in range(total_images):

                cur_image_np = image_np[i,:, :, ::-1]

                original_resolution = cur_image_np.shape[0:2]

                if facerestore_model is None or self.face_helper is None:
                    return result

                self.face_helper.clean_all()
                self.face_helper.read_image(cur_image_np)
                self.face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
                self.face_helper.align_warp_face()

                # Face-Filter Mode

                # Фильтрация лиц
                if face_selection != "all" and self.face_helper.cropped_faces:
                    # Собираем информацию о лицах для фильтрации
                    face_info = []
                    img_height, img_width = cur_image_np.shape[0:2]
                    
                    for j, face in enumerate(self.face_helper.cropped_faces):
                        # Используем центр лица вместо левого верхнего угла
                        if hasattr(self.face_helper, 'det_faces') and len(self.face_helper.det_faces) > j:
                            bbox = self.face_helper.det_faces[j]
                            # Вычисляем центр лица для более точного позиционирования
                            x1 = ((bbox[0] + bbox[2]) / 2) / img_width  # центр x
                            y1 = ((bbox[1] + bbox[3]) / 2) / img_height  # центр y
                            area = face.shape[0] * face.shape[1]
                            confidence = bbox[4] if len(bbox) > 4 else 1.0
                        else:
                            # Если информация о bbox недоступна, используем приблизительные данные
                            area = face.shape[0] * face.shape[1]
                            x1, y1 = 0.5, 0.5  # центр изображения
                            confidence = 1.0
                            
                        face_info.append({
                            'index': j,
                            'area': area,
                            'x_position': x1,
                            'y_position': y1,
                            'detection_confidence': confidence
                        })
                    
                    # Сначала сортируем все лица по выбранному критерию
                    all_indices = list(range(len(self.face_helper.cropped_faces)))
                    
                    # Вывод для x_position и y_position
                    if sort_by == "y_position":
                        all_positions = [(idx, face_info[idx]['y_position']) for idx in all_indices]
                    elif sort_by == "x_position":
                        all_positions = [(idx, face_info[idx]['x_position']) for idx in all_indices]
                    
                    # Сортировка по выбранному критерию
                    sorted_indices = sorted(
                        all_indices,
                        key=lambda idx: face_info[idx][sort_by],
                        reverse=reverse_order
                    )
                    
                    # Отладочный вывод после сортировки
                    if sort_by == "y_position":
                        sorted_positions = [(idx, face_info[idx]['y_position']) for idx in sorted_indices]
                    elif sort_by == "x_position":
                        sorted_positions = [(idx, face_info[idx]['x_position']) for idx in sorted_indices]
                    
                    # Применяем фильтрацию в зависимости от режима
                    if face_selection == "filter":
                        # Фильтрация по координатам
                        filtered_indices = [
                            idx for idx in sorted_indices
                            if min_x_position <= face_info[idx]['x_position'] <= max_x_position and
                               min_y_position <= face_info[idx]['y_position'] <= max_y_position
                        ]
                        
                        # Выборка по take_start и take_count
                        selected_indices = filtered_indices[take_start:take_start + take_count]
                    
                    elif face_selection == "largest":
                        # При выборе "largest" просто берем take_count лиц с наибольшей площадью, начиная с take_start
                        selected_indices = sorted_indices[take_start:take_start + take_count]
                    
                    elif face_selection == "index":
                        # В режиме "index" просто берем лица, начиная с take_start
                        selected_indices = sorted_indices[take_start:take_start + take_count]

                    if selected_indices:
                        self.face_helper.cropped_faces = [self.face_helper.cropped_faces[j] for j in selected_indices]
                        if hasattr(self.face_helper, 'restored_faces'):
                            self.face_helper.restored_faces = []
                        if hasattr(self.face_helper, 'affine_matrices'):
                            self.face_helper.affine_matrices = [self.face_helper.affine_matrices[j] for j in selected_indices]
                        if hasattr(self.face_helper, 'det_faces'):
                            self.face_helper.det_faces = [self.face_helper.det_faces[j] for j in selected_indices]

                # Face-Filter Mode END
                
                restored_face = None

                for idx, cropped_face in enumerate(self.face_helper.cropped_faces):

                    cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
                    normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                    cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

                    try:

                        with torch.no_grad():

                            if ".onnx" in face_restore_model: # ONNX models

                                for ort_session_input in ort_session.get_inputs():
                                    if ort_session_input.name == "input":
                                        cropped_face_prep = prepare_cropped_face(cropped_face)
                                        ort_session_inputs[ort_session_input.name] = cropped_face_prep
                                    if ort_session_input.name == "weight":
                                        weight = np.array([ 1 ], dtype = np.double)
                                        ort_session_inputs[ort_session_input.name] = weight

                                output = ort_session.run(None, ort_session_inputs)[0][0]
                                restored_face = normalize_cropped_face(output)

                            else: # PTH models

                                output = facerestore_model(cropped_face_t, w=codeformer_weight)[0] if "codeformer" in face_restore_model.lower() else facerestore_model(cropped_face_t)[0]
                                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))

                        del output
                        torch.cuda.empty_cache()

                    except Exception as error:

                        print(f"\tFailed inference: {error}", file=sys.stderr)
                        restored_face = cropped_face.copy()

                    if visibility < 1:
                        restored_face = cropped_face * (1 - visibility) + restored_face * visibility

                    restored_face = restored_face.astype("uint8")
                    self.face_helper.add_restored_face(restored_face)

                self.face_helper.get_inverse_affine(None)

                restored_img = self.face_helper.paste_faces_to_input_image()
                restored_img = restored_img[:, :, ::-1]

                if original_resolution != restored_img.shape[0:2]:
                    restored_img = cv2.resize(restored_img, (0, 0), fx=original_resolution[1]/restored_img.shape[1], fy=original_resolution[0]/restored_img.shape[0], interpolation=cv2.INTER_AREA)

                self.face_helper.clean_all()

                out_images.append(restored_img)

                if state.interrupted or model_management.processing_interrupted():
                    logger.status("Interrupted by User")
                    return image
                
                pbar.update(1)

            restored_img_np = np.array(out_images).astype(np.float32) / 255.0
            restored_img_tensor = torch.from_numpy(restored_img_np)

            result = restored_img_tensor

            progress_bar_reset(pbar)

        return (result,)


# Process-wide insightface detector for mask_eyes. ComfyUI re-instantiates nodes
# on every queue, so an instance-level cache would re-create the ORT sessions
# each run (heavy VRAM alloc/dealloc churn — on a full GPU the session init can
# fail with bad allocation and even take the process down).
_FACE_AUTHORITY_CACHE = {}


class MaskHelper:
    def __init__(self):
        self._blur_cache = {}
        self._eye_debug = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "swapped_image": ("IMAGE",),
                # 必需输入：ReActor 的 SWAP_MASK 输出。不连线 = 工作流未建全，
                # ComfyUI 直接标红，不做任何静默兜底（有利于统一链路与排查）。
                "swap_mask": ("MASK",),
                "morphology_operation": (["dilate", "erode", "open", "close"],),
                "morphology_distance": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "blur_radius": ("INT", {"default": 9, "min": 0, "max": 48, "step": 1}),
                "sigma_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 3., "step": 0.01}),
                "mask_eyes": ("BOOLEAN", {"default": False, "label_off": "no", "label_on": "yes"}),
                "eye_size": ("FLOAT", {"default": 1.2, "min": 0.5, "max": 2.5, "step": 0.05}),
                "eye_dilation": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),
                "eye_feather": ("INT", {"default": 12, "min": 0, "max": 64, "step": 1}),
            },
            "optional": {
                "mask_optional": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE","MASK","IMAGE","IMAGE")
    RETURN_NAMES = ("IMAGE","MASK","MASK_PREVIEW","SWAPPED_FACE")
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor"

    def _get_face_authority(self):
        """InsightFace RetinaFace(+1k3d68) detector — the same quality bar ReActor
        itself uses for swapping. Practically junk-free at its default threshold
        (unlike BlazeFace), so no geometric anti-junk heuristics are needed.
        Cached process-wide; falls back to CPU providers when the GPU is too full
        for a new ORT CUDA session (bad allocation) instead of disabling."""
        global _FACE_AUTHORITY_CACHE
        if "app" in _FACE_AUTHORITY_CACHE:
            return _FACE_AUTHORITY_CACHE["app"]
        import insightface
        # insightface_path/providers are module-level imports: ComfyUI's loader
        # removes the custom-node dir from sys.path after import, so a runtime
        # 'from scripts.reactor_swapper import ...' would fail
        app = None
        last_err = None
        for provs in (providers, ["CPUExecutionProvider"]):
            try:
                candidate = insightface.app.FaceAnalysis(
                    name="buffalo_l",
                    # task name differs across insightface versions: 'landmark_3d_68' vs 'landmark_3d68'
                    allowed_modules=["detection", "landmark_3d_68", "landmark_3d68"],
                    providers=provs,
                    root=insightface_path,
                )
                # 略降检测阈值（与换脸路径的 VIDEO_DET_THRESH 一致）：
                # 运动模糊帧漏检 = mask 掉空 = 合成回原脸 = 闪烁
                candidate.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.4)
                app = candidate
                break
            except Exception as e:
                last_err = e
                logger.warning(f"mask_eyes: insightface init failed with providers={provs} ({e})")
        if app is None:
            raise RuntimeError(f"mask_eyes: insightface detector unavailable ({last_err})")
        _FACE_AUTHORITY_CACHE["app"] = app
        return app

    def _eye_entries_from_face(self, face):
        """Convert an insightface Face into a hole-drawing entry. Eye rings come
        from the 68-point landmarks (iBUG layout: 36-41 / 42-47 per eye); when
        landmarks are unavailable a coarse ellipse pair is derived from the
        5-point kps (never draw nothing — a missed eye is a black eye)."""
        kps = getattr(face, "kps", None)
        if kps is None:
            return None
        kps = np.asarray(kps, dtype=np.float32)
        if kps.ndim != 2 or kps.shape[0] < 2 or not np.isfinite(kps).all():
            return None
        e0, e1 = kps[0], kps[1]
        d = float(np.linalg.norm(e0 - e1))
        if d < 4.0:
            return None
        lmk = getattr(face, "landmark_3d_68", None)
        if lmk is None:
            # older insightface versions name it without the extra underscore
            lmk = getattr(face, "landmark_3d68", None)
        if lmk is not None:
            pts = np.asarray(lmk, dtype=np.float32)[:, :2]
            if pts.shape[0] >= 48 and np.isfinite(pts).all():
                return {
                    "bbox_xyxy": np.asarray(face.bbox, dtype=np.float32),
                    "landmarks_xy": pts,
                    "rings": [np.arange(36, 42), np.arange(42, 48)],
                    "score": float(getattr(face, "det_score", 0.0)),
                    "eye_src": "lmk68",
                }
        a, b = 0.26 * d, 0.115 * d
        th = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False, dtype=np.float32)
        shape = np.stack([a * np.cos(th), b * np.sin(th)], axis=1).astype(np.float32)
        lmks = np.concatenate([e0[None, :] + shape, e1[None, :] + shape], axis=0)
        return {
            "bbox_xyxy": np.asarray(face.bbox, dtype=np.float32),
            "landmarks_xy": lmks,
            "rings": [np.arange(0, 16), np.arange(16, 32)],
            "score": float(getattr(face, "det_score", 0.0)),
            "eye_src": "kps",
        }

    def _detect_eye_faces(self, frames, region_mask=None, dbg=False):
        """Per-frame authoritative face detection (RetinaFace) with eye-ring
        geometry. One canonical detection per real face keeps the holes stable.

        Two-stage: boxes first, then landmarks only for faces inside the swap
        region. Per-face 68-point landmark inference (~16ms/face on CPU)
        dominates detection in crowd scenes, and faces outside the swap region
        never need holes — their landmarks are pure waste."""
        app = self._get_face_authority()
        results = []
        for i, frame in enumerate(frames):
            bgr = np.ascontiguousarray(frame[:, :, ::-1])
            # 检测异常直接抛出：静默吞掉只会留下"眼洞没打上"的难查结果。
            # 不传 max_num：insightface 在 max_num>0 时按"面积−2×到图中心距离²"
            # 选脸（center-bias 启发式），人群场景会把远离中心的主体大脸挤掉、
            # 只留中心附近的小脸——主体脸眼洞漏打，眼睛被 GFPGAN 重画。
            # 两段式精确复刻 insightface 0.7.3 FaceAnalysis.get()，只是把
            # 68 点关键点推理推迟到区域过滤之后。
            bboxes, kpss = app.det_model.detect(bgr, max_num=0, metric="default")
            frame_region = None
            if region_mask is not None:
                frame_region = (region_mask[i] > 0.5).cpu().numpy()
            entries = []
            for j in range(bboxes.shape[0]):
                bbox = bboxes[j, 0:4]
                if frame_region is not None:
                    # bbox 外扩 8px 后与换脸区域求交：无交集的脸不换、不需要眼洞
                    x1, y1 = max(int(bbox[0]) - 8, 0), max(int(bbox[1]) - 8, 0)
                    x2 = min(int(bbox[2]) + 8, frame_region.shape[1])
                    y2 = min(int(bbox[3]) + 8, frame_region.shape[0])
                    if x2 <= x1 or y2 <= y1 or not frame_region[y1:y2, x1:x2].any():
                        continue
                kps = kpss[j] if kpss is not None else None
                face = Face(bbox=bbox, kps=kps, det_score=bboxes[j, 4])
                for taskname, model in app.models.items():
                    if taskname == "detection":
                        continue
                    model.get(bgr, face)
                e = self._eye_entries_from_face(face)
                if e is not None:
                    entries.append(e)
            results.append(entries)
        self._temporal_smooth(results, dbg=dbg, src="auth")
        self._coast_missing_faces(results, dbg=dbg)
        if dbg:
            for i, faces in enumerate(results):
                if not faces:
                    logger.status(f"mask_eyes[dbg] auth f{i}: 0 faces")
                for f in faces:
                    b = f["bbox_xyxy"]
                    eyes = " ".join(
                        f"({p.mean(axis=0)[0]:.0f},{p.mean(axis=0)[1]:.0f})r{np.linalg.norm(p.max(axis=0) - p.min(axis=0)) / 2:.0f}"
                        for p in (f["landmarks_xy"][ring].astype(np.float32) for ring in f["rings"]))
                    logger.status(
                        f"mask_eyes[dbg] auth f{i}: src={f['eye_src']} score={f['score']:.2f} "
                        f"box=({b[0]:.0f},{b[1]:.0f})-({b[2]:.0f},{b[3]:.0f}) eyes={eyes}")
        return results

    def _temporal_smooth(self, frames_faces, dbg=False, src="auth", alpha=0.5):
        """Per-face EMA on the eye-ring centroid/radius. A single-frame landmark
        glitch (a ring jumping to the mouth/cheek) is held at the previous position:
        when the raw centroid jumps more than 0.3x face size from the track history,
        the history value is kept (alpha=0) until the raw detection returns."""
        tracks = []  # {"center": (x, y), "size": s, "rings": [(cx, cy, r), ...]}
        for fi, faces in enumerate(frames_faces):
            used = set()
            assignments = []
            for f in faces:
                b = f["bbox_xyxy"]
                c = np.array([(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5], dtype=np.float32)
                size = max(float(b[2] - b[0]), float(b[3] - b[1]), 1.0)
                best_t, best_d = None, None
                for ti, t in enumerate(tracks):
                    if ti in used:
                        continue
                    d = float(math.hypot(c[0] - t["center"][0], c[1] - t["center"][1]))
                    if d <= 0.5 * max(size, t["size"]) and (best_d is None or d < best_d):
                        best_t, best_d = ti, d
                assignments.append((f, c, size, best_t))
                if best_t is not None:
                    used.add(best_t)
            for f, c, size, ti in assignments:
                raw = []
                for ring in f["rings"]:
                    pts = f["landmarks_xy"][ring].astype(np.float32)
                    ctr = pts.mean(axis=0)
                    rad = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) * 0.5
                    raw.append((float(ctr[0]), float(ctr[1]), rad))
                if ti is None:
                    track = {"center": (float(c[0]), float(c[1])), "size": size, "rings": raw}
                    tracks.append(track)
                else:
                    track = tracks[ti]
                    smoothed = []
                    for ri, (r, p) in enumerate(zip(raw, track["rings"])):
                        jump = math.hypot(r[0] - p[0], r[1] - p[1])
                        w = alpha if jump <= 0.3 * size else 0.0
                        if dbg and w == 0.0:
                            logger.status(f"mask_eyes[dbg] {src} f{fi}: ring{ri} jump {jump:.0f}px > {0.3 * size:.0f}px, held at ({p[0]:.0f},{p[1]:.0f})r{p[2]:.0f}")
                        smoothed.append((w * r[0] + (1 - w) * p[0],
                                         w * r[1] + (1 - w) * p[1],
                                         w * r[2] + (1 - w) * p[2]))
                    track["rings"] = smoothed
                    track["center"] = (float(c[0]), float(c[1]))
                    track["size"] = size
                # Write the smoothed geometry back into the landmarks
                lmks = f["landmarks_xy"]
                for ring, (scx, scy, srad) in zip(f["rings"], track["rings"]):
                    pts = lmks[ring].astype(np.float32)
                    ctr = pts.mean(axis=0)
                    rad = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) * 0.5
                    if rad > 0:
                        lmks[ring] = np.array([scx, scy], dtype=np.float32) + (pts - ctr) * (srad / rad)

    def _coast_missing_faces(self, frames_faces, coast_limit=10, dbg=False):
        """短暂漏检桥接（coast）。手遮挡后大幅运动/快速转头时，逐帧检测会
        间歇性漏掉主脸：漏检帧没有 face 条目 → 该帧 mask/眼洞缺失 →
        合成结果整帧回退为原图，表现为"换脸/原脸"快速交替闪烁。
        这里按时序把人脸锁定到轨迹上，漏检帧沿用该轨迹最近一次的
        bbox+眼环数据（最多 coast_limit 帧），让 mask 保持稳定。"""
        tracks = []  # {"entry": 最近条目, "center": (x, y), "size": s, "miss": 连续丢失帧数}
        for fi, faces in enumerate(frames_faces):
            used = set()
            for f in faces:
                b = f["bbox_xyxy"]
                c = np.array([(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5], dtype=np.float32)
                size = max(float(b[2] - b[0]), float(b[3] - b[1]), 1.0)
                best_t, best_d = None, None
                for ti, t in enumerate(tracks):
                    if ti in used:
                        continue
                    d = float(math.hypot(c[0] - t["center"][0], c[1] - t["center"][1]))
                    if d <= 0.6 * max(size, t["size"]) and (best_d is None or d < best_d):
                        best_t, best_d = ti, d
                if best_t is not None:
                    used.add(best_t)
                    t = tracks[best_t]
                    t["entry"] = f
                    t["center"] = (float(c[0]), float(c[1]))
                    t["size"] = size
                    t["miss"] = 0
                else:
                    tracks.append({"entry": f, "center": (float(c[0]), float(c[1])), "size": size, "miss": 0})
                    used.add(len(tracks) - 1)
            for ti, t in enumerate(tracks):
                if ti in used:
                    continue
                t["miss"] += 1
                if t["miss"] <= coast_limit:
                    # 沿用最近一次检测到的框/眼环（浅拷贝即可，下游只读）
                    faces.append(dict(t["entry"]))
        return frames_faces

    @staticmethod
    def _to_uint8_frames(t):
        t = t[..., :3]
        if t.ndim == 3:
            t = t.unsqueeze(0)
        return t.clamp(0, 1).mul(255.0).add(0.5).to(torch.uint8).cpu().numpy()

    def _build_eye_mask(self, image, eye_size, eye_dilation, eye_feather=12, region_mask=None):
        """Punch eye holes into the swap mask. Faces come from the authoritative
        RetinaFace detector (junk-free) with eye rings from its 68-point landmarks
        (kps-ellipse fallback). Two-scale holes: a solid core over the eye opening
        plus a wide soft outer ramp. Returns a (B, H, W) float tensor or None."""
        dbg = self._eye_debug
        img = image if image.ndim == 4 else image.unsqueeze(0)
        B, H, W = img.shape[0], img.shape[1], img.shape[2]
        faces_per_frame = self._detect_eye_faces(list(self._to_uint8_frames(img)), region_mask=region_mask, dbg=dbg)
        return self._eye_mask_from_faces(faces_per_frame, B, H, W, eye_size, eye_dilation, eye_feather, dbg=dbg, region_mask=region_mask)

    def _eye_mask_from_faces(self, faces_per_frame, B, H, W, eye_size, eye_dilation, eye_feather=12, dbg=False, region_mask=None):
        """Draw the two-scale eye holes (solid core + feathered outer ramp) for the
        given per-frame face entries. Returns a (B, H, W) float tensor.

        绘制局部化：逐脸全帧 GaussianBlur 是眼洞绘制的大头（768x1344 每脸
        ~13ms）。该脸的 outer mask 在多边形支撑域外本来就是 0，所以在"多边形
        外扩模糊核半径"的局部裁剪里画+补零模糊（BORDER_CONSTANT 与全帧逐位
        等价），再整块贴回；裁剪触及图像边界时回退全帧（reflect 边界行为
        与补零不等价）。"""
        masks = []
        failed_frames = []
        for i in range(B):
            core_m = np.zeros((H, W), dtype=np.float32)
            outer_acc = np.zeros((H, W), dtype=np.float32)
            per_frame = faces_per_frame[i]
            rings_drawn = 0
            for face in per_frame:
                lmks = face["landmarks_xy"]
                x1, y1, x2, y2 = face["bbox_xyxy"]
                face_w = max(float(x2 - x1), 1.0)
                ring_pts = [lmks[ring].astype(np.float32) for ring in face["rings"]]
                centers = [p.mean(axis=0) for p in ring_pts]
                if len(centers) == 2:
                    d = float(np.linalg.norm(centers[0] - centers[1]))
                    # Implausible eye pair (inter-ocular distance / vertical alignment)
                    if not (0.08 * face_w <= d <= 0.75 * face_w) or abs(centers[0][1] - centers[1][1]) > 0.5 * d:
                        if dbg:
                            logger.status(f"mask_eyes[dbg] f{i}: implausible eye pair (d={d:.0f}px, face_w={face_w:.0f}px), face skipped")
                        continue
                for ri, (pts, center) in enumerate(zip(ring_pts, centers)):
                    ring_diam = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
                    # Skip degenerate rings (landmarks can collapse on hard poses)
                    if not (0.04 * face_w <= ring_diam <= 0.65 * face_w):
                        if dbg:
                            logger.status(f"mask_eyes[dbg] f{i}: ring{ri} degenerate (diam={ring_diam:.0f}px, face_w={face_w:.0f}px), skipped")
                        continue
                    if not (x1 - 0.25 * face_w <= center[0] <= x2 + 0.25 * face_w and
                            y1 - 0.25 * face_w <= center[1] <= y2 + 0.25 * face_w):
                        if dbg:
                            logger.status(f"mask_eyes[dbg] f{i}: ring{ri} outside face box (center=({center[0]:.0f},{center[1]:.0f})), skipped")
                        continue
                    # Two-scale hole: a SOLID core over the eye opening (guarantees
                    # 100% original pixels there — full iris colour and sharp
                    # catchlights) plus a soft outer ramp for the tone transition.
                    # A single blurred polygon never saturates on small faces
                    # (feather sigma > hole radius), leaving the eye half-swapped.
                    rad = float(np.mean(np.linalg.norm(pts - center, axis=1)))
                    margin = float(eye_dilation) * min(max(face_w / 256.0, 0.25), 3.0)
                    core_pts = center + (pts - center) * float(eye_size)
                    outer_pts = center + (pts - center) * (float(eye_size) + margin / max(rad, 1.0))
                    cv2.fillPoly(core_m, [np.round(core_pts).astype(np.int32)], 1.0)
                    rings_drawn += 1
                    # 羽化半径按人脸宽度自适应：固定 sigma 在小脸上（如 35px）会把
                    # 眼洞羽化扩散到整张脸，(1-eye) 大面积压低换脸权重，看起来像
                    # "没有换脸"。sigma = min(eye_feather, 0.12*face_w)：
                    # face_w>=100px 时与旧的全局行为完全一致，小脸则按比例收缩。
                    sigma = float(min(eye_feather, max(0.12 * face_w, 1.0)))
                    if eye_feather > 0 and sigma > 0:
                        # 绘制局部化：该脸 outer mask 在多边形支撑域外本来就是 0，
                        # 在"多边形外扩模糊核半径"的局部裁剪里画+补零模糊
                        # （BORDER_CONSTANT 与全帧逐位等价）再贴回。OpenCV float32
                        # 自动核尺寸 ≈ 8σ+1（半径 ~4σ），外扩取 4σ+2 宁大勿小。
                        # 裁剪触及图像边界时回退全帧：全帧模糊在图像边界是
                        # reflect 模式，与补零不等价（眼部贴边的极端构图）。
                        krad = int(4 * sigma) + 2
                        px0 = int(np.floor(float(outer_pts[:, 0].min()))) - krad
                        py0 = int(np.floor(float(outer_pts[:, 1].min()))) - krad
                        px1 = int(np.ceil(float(outer_pts[:, 0].max()))) + krad + 1
                        py1 = int(np.ceil(float(outer_pts[:, 1].max()))) + krad + 1
                        if px0 < 0 or py0 < 0 or px1 > W or py1 > H:
                            face_outer = np.zeros((H, W), dtype=np.float32)
                            cv2.fillPoly(face_outer, [np.round(outer_pts).astype(np.int32)], 1.0)
                            face_outer = cv2.GaussianBlur(face_outer, (0, 0), sigmaX=sigma)
                            mx = float(face_outer.max())
                            if mx > 0:
                                face_outer = face_outer / mx
                            np.maximum(outer_acc, face_outer, out=outer_acc)
                        else:
                            face_outer = np.zeros((py1 - py0, px1 - px0), dtype=np.float32)
                            cv2.fillPoly(face_outer, [np.round(outer_pts).astype(np.int32) - (px0, py0)], 1.0)
                            face_outer = cv2.GaussianBlur(face_outer, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_CONSTANT)
                            mx = float(face_outer.max())
                            if mx > 0:
                                face_outer = face_outer / mx
                            np.maximum(outer_acc[py0:py1, px0:px1], face_outer, out=outer_acc[py0:py1, px0:px1])
                    else:
                        cv2.fillPoly(outer_acc, [np.round(outer_pts).astype(np.int32)], 1.0)
            m = np.maximum(core_m, outer_acc)
            if region_mask is not None:
                region_present = bool((region_mask[i] > 0.5).any())
            else:
                region_present = True
            if rings_drawn == 0 and (per_frame or region_present):
                # 眼洞为空且确实有东西要保护：脸已检测到但关键点退化，或换脸
                # 区域匹配不到任何脸——该帧输出=完全换后脸。必须显式报告，
                # 不能伪装成成功。区域缺席的帧（人脸尚未入镜）不算失败。
                failed_frames.append(i)
            masks.append(torch.from_numpy(m))
        if failed_frames:
            shown = str(failed_frames[:10]).rstrip(']')
            suffix = ", ..." if len(failed_frames) > 10 else "]"
            logger.warning(f"mask_eyes: no usable eyes on frame(s) {shown}{suffix} out of {B} — those frames keep the FULLY SWAPPED face (no eye preservation)")
        else:
            logger.status(f"mask_eyes: eye regions excluded from the swap mask in all {B} frames")
        return torch.stack(masks)

    def execute(self, image, swapped_image, swap_mask, morphology_operation, morphology_distance, blur_radius, sigma_factor, mask_eyes, eye_size, eye_dilation, eye_feather, mask_optional=None):
        device = model_management.get_torch_device()
        # self._eye_debug: code-level constant (see __init__), not exposed in the UI

        # ВАЖНО: не перемещаем image/swapped_image на GPU целиком. Для видео (сотни
        # кадров) полный fp32-батч занимает гигабайты VRAM. Батчи остаются на CPU —
        # на GPU попадают только погранные срезы при композитинге (fast path ниже)
        # или весь батч в generic path.

        # 区域来源（显式链路，无静默兜底）：
        # swap_mask ← ReActor 的 SWAP_MASK 输出（必需，继承前节点已算好的换脸区域）；
        # mask_optional ← 用户外接自定义 mask，连接时优先（显式覆盖）。
        combined_mask = mask_optional if mask_optional is not None else swap_mask
        if (not isinstance(combined_mask, torch.Tensor)) or combined_mask.shape[0] != image.shape[0] or tuple(combined_mask.shape[1:3]) != tuple(image.shape[1:3]):
            raise RuntimeError(
                f"MaskHelper: region mask shape {getattr(combined_mask, 'shape', type(combined_mask).__name__)} "
                f"doesn't match image {tuple(image.shape)} — check the SWAP_MASK / mask_optional connection"
            )
        if combined_mask.dtype != torch.float32:
            combined_mask = combined_mask.float()

        # Маски обрабатываем на compute device: morph/eye/blur покадровые и не
        # зависят от устройства.
        if isinstance(combined_mask, torch.Tensor) and combined_mask.device != device:
            combined_mask = combined_mask.to(device)

        # Morph operations
        if morphology_operation == "dilate":
            # print(f"max before: {combined_mask.max()}, min: {combined_mask.min()}, sum: {combined_mask.sum()}")
            combined_mask = self.iterative_morphology(combined_mask, morphology_distance, op="dilate")
            # print(f"after morph: {combined_mask.max()}, min: {combined_mask.min()}, sum: {combined_mask.sum()}")
        elif morphology_operation == "erode":
            combined_mask = self.iterative_morphology(combined_mask, morphology_distance, op="erode")
        elif morphology_operation == "open":
            combined_mask = self.iterative_morphology(self.iterative_morphology(combined_mask, morphology_distance, op="erode"), morphology_distance, op="dilate")
        elif morphology_operation == "close":
            combined_mask = self.iterative_morphology(self.iterative_morphology(combined_mask, morphology_distance, op="dilate"), morphology_distance, op="erode")

        # Preserve original eyes: exclude eye regions from the swap mask (authoritative RetinaFace detection)
        if mask_eyes:
            eye_mask = self._build_eye_mask(image, eye_size, eye_dilation, eye_feather, region_mask=combined_mask)
            eye_mask = eye_mask.to(device=combined_mask.device, dtype=combined_mask.dtype)
            if eye_mask.shape[0] != combined_mask.shape[0]:
                raise RuntimeError(f"mask_eyes: eye batch {eye_mask.shape[0]} doesn't match mask batch {combined_mask.shape[0]}")
            combined_mask = combined_mask * (1.0 - eye_mask)

        # Gaussian blur
        if blur_radius > 0:
            blur_key = f"{blur_radius}_{sigma_factor}"
            if blur_key not in self._blur_cache:
                self._blur_cache[blur_key] = T.GaussianBlur(kernel_size=blur_radius * 2 + 1, sigma=sigma_factor)
            blur = self._blur_cache[blur_key]
            mask_blurred = blur(combined_mask.unsqueeze(1)).squeeze(1)
        else:
            mask_blurred = combined_mask

        # Apply mask to swapped image (basic RGBA composite)
        # NOTE: swapped_image остаётся на CPU — fast path композитит погранно,
        # generic path ниже сам переносит нужные тензоры на GPU.

        mask_image_final = mask_blurred

        # *** FAST PATH: aligned batch (standard video use-case) ***
        # swapped_image is the same frames with the face swapped, perfectly aligned
        # to the original — composite directly. The generic cut/resize/paste path
        # below rescales EVERY face to the largest bbox in the batch, distorting
        # smaller faces and making the result flicker frame-to-frame.
        #
        # Composite chunk-by-chunk: a 243-frame 720p batch converted to RGBA in one
        # go needs ~4 GB per tensor (several of them simultaneously) and OOMs.
        if swapped_image.shape[0] == image.shape[0] and tuple(swapped_image.shape[1:3]) == tuple(image.shape[1:3]):
            mask0 = core.tensor2mask(mask_image_final)
            H0, W0 = int(image.shape[1]), int(image.shape[2])
            mask0 = torch.nn.functional.interpolate(mask0.unsqueeze(1), size=(H0, W0), mode='nearest')[:, 0, :, :]
            MB0 = mask0.shape[0]
            if MB0 < image.shape[0]:
                mask0 = mask0.repeat(image.shape[0] // MB0, 1, 1)
            # Полные маски переносим на CPU: во время композитинга на GPU держится
            # только текущий чанк (~1-2 GB), иначе длинное видео исчерпывает VRAM.
            if mask0.device.type != 'cpu':
                mask0 = mask0.cpu()
            if combined_mask.device.type != 'cpu':
                combined_mask = combined_mask.cpu()
            if mask_blurred.device.type != 'cpu':
                mask_blurred = mask_blurred.cpu()
            mask_image_final = mask_blurred  # отпускаем GPU-копию маски

            C0 = int(image.shape[3])

            def _composite_chunk(sl, use_gpu):
                img_c = core.tensor2rgba(image[sl])
                swp_c = core.tensor2rgba(swapped_image[sl])
                mk_c = mask0[sl].unsqueeze(-1)
                if use_gpu:
                    img_c = img_c.to(device)
                    swp_c = swp_c.to(device)
                    mk_c = mk_c.to(device)
                comp = torch.lerp(img_c, swp_c, mk_c)
                del img_c, swp_c
                seg = comp.clone()
                seg[..., 3] = mk_c.squeeze(-1)
                rgb = core.tensor2rgb(comp) if C0 == 3 else comp
                del comp
                return rgb.cpu(), seg.cpu()

            results = []
            segments = []
            chunk = 32
            for i in range(0, int(image.shape[0]), chunk):
                sl = slice(i, min(i + chunk, int(image.shape[0])))
                try:
                    rgb, seg = _composite_chunk(sl, use_gpu=True)
                except RuntimeError as e:
                    logger.warning(f"GPU composite failed on chunk {i} ({e}); retrying on CPU")
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    rgb, seg = _composite_chunk(sl, use_gpu=False)
                results.append(rgb)
                segments.append(seg)
                del rgb, seg
            result = torch.cat(results, dim=0)
            face_segment = torch.cat(segments, dim=0)
            return (result, combined_mask, mask_blurred, face_segment)
        # *** generic path below for non-aligned inputs ***

        # Generic path перемешивает image/swapped/mask в общих операциях и ожидает
        # их на compute device — переносим полный батч (как раньше, до погранного
        # fast path). Сюда попадают только несовпадающие по размеру входы.
        image = image.to(device) if image.device != device else image
        swapped_image = swapped_image.to(device) if swapped_image.device != device else swapped_image
        if mask_image_final.device != device:
            mask_image_final = mask_image_final.to(device)

        # *** CUT BY MASK ***:
    
        if len(swapped_image.shape) < 4:
            C = 1
        else:
            C = swapped_image.shape[3]

        # We operate on RGBA to keep the code clean and then convert back after
        swapped_image = core.tensor2rgba(swapped_image)
        mask = core.tensor2mask(mask_image_final)

        # Scale the mask to be a matching size if it isn't
        B, H, W, _ = swapped_image.shape
        mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest')[:,0,:,:]
        MB, _, _ = mask.shape

        if MB < B:
            assert(B % MB == 0)
            mask = mask.repeat(B // MB, 1, 1)

        # masks_to_boxes errors if the tensor is all zeros, so we'll add a single pixel and zero it out at the end
        is_empty = ~torch.gt(torch.max(torch.reshape(mask,[MB, H * W]), dim=1).values, 0.)
        mask[is_empty,0,0] = 1.
        boxes = masks_to_boxes(mask)
        mask[is_empty,0,0] = 0.

        min_x = boxes[:,0]
        min_y = boxes[:,1]
        max_x = boxes[:,2]
        max_y = boxes[:,3]

        width = max_x - min_x + 1
        height = max_y - min_y + 1

        use_width = int(torch.max(width).item())
        use_height = int(torch.max(height).item())

        alpha_mask = torch.ones((B, H, W, 4), device=device)
        alpha_mask[:,:,:,3] = mask

        swapped_image = swapped_image * alpha_mask

        cutted_image = torch.zeros((B, use_height, use_width, 4), device=device)
        for i in range(0, B):
            if not is_empty[i]:
                ymin = int(min_y[i].item())
                ymax = int(max_y[i].item())
                xmin = int(min_x[i].item())
                xmax = int(max_x[i].item())
                single = (swapped_image[i, ymin:ymax+1, xmin:xmax+1,:]).unsqueeze(0)
                resized = torch.nn.functional.interpolate(single.permute(0, 3, 1, 2), size=(use_height, use_width), mode='bicubic').permute(0, 2, 3, 1)
                cutted_image[i] = resized[0]
        
        # Preserve our type unless we were previously RGB and added non-opaque alpha due to the mask size
        if C == 1:
            cutted_image = core.tensor2mask(cutted_image)
        elif C == 3 and torch.min(cutted_image[:,:,:,3]) == 1:
            cutted_image = core.tensor2rgb(cutted_image)

        # *** PASTE BY MASK ***:

        image_base = core.tensor2rgba(image)
        image_to_paste = core.tensor2rgba(cutted_image)
        mask = core.tensor2mask(mask_image_final)

        # Scale the mask to be a matching size if it isn't
        B, H, W, C = image_base.shape
        MB = mask.shape[0]
        PB = image_to_paste.shape[0]

        if B < PB:
            assert(PB % B == 0)
            image_base = image_base.repeat(PB // B, 1, 1, 1)
        B, H, W, C = image_base.shape
        if MB < B:
            assert(B % MB == 0)
            mask = mask.repeat(B // MB, 1, 1)
        elif B < MB:
            assert(MB % B == 0)
            image_base = image_base.repeat(MB // B, 1, 1, 1)
        if PB < B:
            assert(B % PB == 0)
            image_to_paste = image_to_paste.repeat(B // PB, 1, 1, 1)

        mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest')[:,0,:,:]
        MB, MH, MW = mask.shape

        # masks_to_boxes errors if the tensor is all zeros, so we'll add a single pixel and zero it out at the end
        is_empty = ~torch.gt(torch.max(torch.reshape(mask,[MB, MH * MW]), dim=1).values, 0.)
        mask[is_empty,0,0] = 1.
        boxes = masks_to_boxes(mask)
        mask[is_empty,0,0] = 0.

        min_x = boxes[:,0]
        min_y = boxes[:,1]
        max_x = boxes[:,2]
        max_y = boxes[:,3]
        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2

        target_width = max_x - min_x + 1
        target_height = max_y - min_y + 1

        result = image_base.detach().clone()
        face_segment = mask_image_final
        
        pbar = progress_bar(MB)
        
        for i in range(0, MB):
            if is_empty[i]:
                pbar.update(1)
                continue
            else:
                image_index = i
                SB, SH, SW, _ = image_to_paste.shape

                # Figure out the desired size
                width = int(target_width[i].item())
                height = int(target_height[i].item())

                width = SW
                height = SH

                # Resize the image we're pasting if needed
                resized_image = image_to_paste[i].unsqueeze(0)

                pasting = torch.ones([H, W, C], device=device)
                ymid = float(mid_y[i].item())
                ymin = int(math.floor(ymid - height / 2)) + 1
                ymax = int(math.floor(ymid + height / 2)) + 1
                xmid = float(mid_x[i].item())
                xmin = int(math.floor(xmid - width / 2)) + 1
                xmax = int(math.floor(xmid + width / 2)) + 1

                _, source_ymax, source_xmax, _ = resized_image.shape
                source_ymin, source_xmin = 0, 0

                if xmin < 0:
                    source_xmin = abs(xmin)
                    xmin = 0
                if ymin < 0:
                    source_ymin = abs(ymin)
                    ymin = 0
                if xmax > W:
                    source_xmax -= (xmax - W)
                    xmax = W
                if ymax > H:
                    source_ymax -= (ymax - H)
                    ymax = H

                pasting[ymin:ymax, xmin:xmax, :] = resized_image[0, source_ymin:source_ymax, source_xmin:source_xmax, :]
                pasting[:, :, 3] = 1.

                pasting_alpha = torch.zeros([H, W], device=device)
                pasting_alpha[ymin:ymax, xmin:xmax] = resized_image[0, source_ymin:source_ymax, source_xmin:source_xmax, 3]

                paste_mask = torch.min(pasting_alpha, mask[i]).unsqueeze(2).repeat(1, 1, 4)

                result[image_index] = pasting * paste_mask + result[image_index] * (1. - paste_mask)

                pbar.update(1)

        # Per-frame alpha preview + RGBA→RGB conversion must happen AFTER the loop:
        # converting inside the loop breaks the next iteration (result becomes 3ch
        # while paste_mask stays 4ch) — this is why batch inputs used to crash here
        face_segment = result
        face_segment[..., 3] = mask

        result = rgba2rgb_tensor(result)
        result = result.cpu()  # Перемещаем результат обратно на CPU

        try:
            torch.cuda.empty_cache()
        except:
            pass

        progress_bar_reset(pbar)
        
        return (result, combined_mask, mask_blurred, face_segment)

    def iterative_morphology(self, image, distance, op="dilate"):
        if distance <= 0:
            return image
        image = image.unsqueeze(1)  # shape [B, 1, H, W] or [1, 1, H, W]
        kernel_size = 2 * distance + 1
        padding = distance
        if op == "dilate":
            image = F.max_pool2d(image, kernel_size=kernel_size, stride=1, padding=padding)
        elif op == "erode":
            image = -F.max_pool2d(-image, kernel_size=kernel_size, stride=1, padding=padding)
        return image.squeeze(1)


class ImageDublicator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "count": ("INT", {"default": 1, "min": 0}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGES",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor"

    def execute(self, image, count):
        images = [image for i in range(count)]
        return (images,)


class ImageRGBA2RGB:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor"

    def execute(self, image):
        out = rgba2rgb_tensor(image)
        return (out,)


class MakeFaceModelBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "face_model1": ("FACE_MODEL",),
            },
            "optional": {
                "face_model2": ("FACE_MODEL",),
                "face_model3": ("FACE_MODEL",),
                "face_model4": ("FACE_MODEL",),
                "face_model5": ("FACE_MODEL",),
                "face_model6": ("FACE_MODEL",),
                "face_model7": ("FACE_MODEL",),
                "face_model8": ("FACE_MODEL",),
                "face_model9": ("FACE_MODEL",),
                "face_model10": ("FACE_MODEL",),
            },
        }

    RETURN_TYPES = ("FACE_MODEL",)
    RETURN_NAMES = ("FACE_MODELS",)
    FUNCTION = "execute"

    CATEGORY = "🌌 ReActor"

    def execute(self, **kwargs):
        if len(kwargs) > 0:
            face_models = [value for value in kwargs.values()]
            return (face_models,)
        else:
            logger.error("Please provide at least 1 `face_model`")
            return (None,)


class ReActorOptions:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_faces_order": (
                    ["left-right","right-left","top-bottom","bottom-top","small-large","large-small"], {"default": "large-small"}
                ),
                "input_faces_index": ("STRING", {"default": "0"}),
                "detect_gender_input": (["no","female","male"], {"default": "no"}),
                "source_faces_order": (
                    ["left-right","right-left","top-bottom","bottom-top","small-large","large-small"], {"default": "large-small"}
                ),
                "source_faces_index": ("STRING", {"default": "0"}),
                "detect_gender_source": (["no","female","male"], {"default": "no"}),
                "console_log_level": ([0, 1, 2], {"default": 1}),
                "restore_swapped_only": ("BOOLEAN", {"default": False, "label_off": "no", "label_on": "yes"})
            }
        }

    RETURN_TYPES = ("OPTIONS",)
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor"

    def execute(self,input_faces_order, input_faces_index, detect_gender_input, source_faces_order, source_faces_index, detect_gender_source, console_log_level, restore_swapped_only):
        options: dict = {
            "input_faces_order": input_faces_order,
            "input_faces_index": input_faces_index,
            "detect_gender_input": detect_gender_input,
            "source_faces_order": source_faces_order,
            "source_faces_index": source_faces_index,
            "detect_gender_source": detect_gender_source,
            "console_log_level": console_log_level,
            "restore_swapped_only": restore_swapped_only,
        }
        return (options, )


class ReActorFaceBoost:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True, "label_off": "OFF", "label_on": "ON"}),
                "boost_model": (get_model_names(get_restorers),),
                "interpolation": (["Nearest","Bilinear","Bicubic","Lanczos"], {"default": "Bicubic"}),
                "visibility": ("FLOAT", {"default": 1, "min": 0.1, "max": 1, "step": 0.05}),
                "codeformer_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1, "step": 0.05}),
                "restore_with_main_after": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("FACE_BOOST",)
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor"

    def execute(self,enabled,boost_model,interpolation,visibility,codeformer_weight,restore_with_main_after):
        face_boost: dict = {
            "enabled": enabled,
            "boost_model": boost_model,
            "interpolation": interpolation,
            "visibility": visibility,
            "codeformer_weight": codeformer_weight,
            "restore_with_main_after": restore_with_main_after,
        }
        return (face_boost, )

class ReActorUnload:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trigger": ("IMAGE", ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor"

    def execute(self, trigger):
        unload_all_models()
        return (trigger,)


NODE_CLASS_MAPPINGS = {
    # --- MAIN NODES ---
    "ReActorFaceSwap": reactor,
    "ReActorFaceSwapOpt": ReActorPlusOpt,
    "ReActorFaceSwapOptWithDirection": ReActorPlusOptWithDirection,
    "ReActorOptions": ReActorOptions,
    "ReActorFaceBoost": ReActorFaceBoost,
    "ReActorMaskHelper": MaskHelper,
    "ReActorSetWeight": ReActorWeight,
    # --- Operations with Face Models ---
    "ReActorSaveFaceModel": SaveFaceModel,
    "ReActorLoadFaceModel": LoadFaceModel,
    "ReActorBuildFaceModel": BuildFaceModel,
    "ReActorMakeFaceModelBatch": MakeFaceModelBatch,
    # --- Additional Nodes ---
    "ReActorRestoreFace": RestoreFace,
    "ReActorRestoreFaceAdvanced": RestoreFaceAdvanced,
    "ReActorImageDublicator": ImageDublicator,
    "ImageRGBA2RGB": ImageRGBA2RGB,
    "ReActorUnload": ReActorUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # --- MAIN NODES ---
    "ReActorFaceSwap": "ReActor 🌌 Fast Face Swap",
    "ReActorFaceSwapOpt": "ReActor 🌌 Fast Face Swap [OPTIONS]",
    "ReActorFaceSwapOptWithDirection": "ReActor 🌌 Fast Face Swap [OPTIONS + DIRECTION]",
    "ReActorOptions": "ReActor 🌌 Options",
    "ReActorFaceBoost": "ReActor 🌌 Face Booster",
    "ReActorMaskHelper": "ReActor 🌌 Masking Helper",
    "ReActorSetWeight": "ReActor 🌌 Set Face Swap Weight",
    # --- Operations with Face Models ---
    "ReActorSaveFaceModel": "Save Face Model 🌌 ReActor",
    "ReActorLoadFaceModel": "Load Face Model 🌌 ReActor",
    "ReActorBuildFaceModel": "Build Blended Face Model 🌌 ReActor",
    "ReActorMakeFaceModelBatch": "Make Face Model Batch 🌌 ReActor",
    # --- Additional Nodes ---
    "ReActorRestoreFace": "Restore Face 🌌 ReActor",
    "ReActorRestoreFaceAdvanced": "Restore Face Advanced 🌌 ReActor",
    "ReActorImageDublicator": "Image Dublicator (List) 🌌 ReActor",
    "ImageRGBA2RGB": "Convert RGBA to RGB 🌌 ReActor",
    "ReActorUnload": "Unload ReActor Models 🌌 ReActor",
}
