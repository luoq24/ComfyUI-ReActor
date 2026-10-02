import os
import shutil
from typing import List, Union

import cv2
import numpy as np
from PIL import Image

import onnxruntime as ort

import insightface
from insightface.app.common import Face
import torch

import folder_paths
import comfy.model_management as model_management
from modules.shared import state

from scripts.reactor_logger import logger
from reactor_utils import (
    move_path,
    get_image_md5hash,
    progress_bar,
    progress_bar_reset
)
from scripts.r_faceboost import swapper, restorer

import warnings

np.warnings = warnings
np.warnings.filterwarnings('ignore')

# PROVIDERS
try:
    if torch.cuda.is_available():
        providers = ["CUDAExecutionProvider"]
    elif torch.backends.mps.is_available():
        providers = ["CoreMLExecutionProvider"]
    elif hasattr(torch,'dml') or hasattr(torch,'privateuseone'):
        providers = ["ROCMExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
except Exception as e:
    logger.debug(f"ExecutionProviderError: {e}.\nEP is set to CPU.")
    providers = ["CPUExecutionProvider"]

models_path_old = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
insightface_path_old = os.path.join(models_path_old, "insightface")
insightface_models_path_old = os.path.join(insightface_path_old, "models")

models_path = folder_paths.models_dir
insightface_path = os.path.join(models_path, "insightface")
insightface_models_path = os.path.join(insightface_path, "models")
reswapper_path = os.path.join(models_path, "reswapper")
hyperswap_path = os.path.join(models_path, "hyperswap")

if os.path.exists(models_path_old):
    move_path(insightface_models_path_old, insightface_models_path)
    move_path(insightface_path_old, insightface_path)
    move_path(models_path_old, models_path)
if os.path.exists(insightface_path) and os.path.exists(insightface_path_old):
    shutil.rmtree(insightface_path_old)
    shutil.rmtree(models_path_old)


FS_MODEL = None
CURRENT_FS_MODEL_PATH = None

ANALYSIS_MODELS = {
    "640": None,
    "320": None,
}

SOURCE_FACES = None
SOURCE_IMAGE_HASH = None
TARGET_FACES = None
TARGET_IMAGE_HASH = None
TARGET_FACES_LIST = []
TARGET_IMAGE_LIST_HASH = []

# 视频逐帧处理时略微降低检测阈值：减少运动模糊/侧脸帧的漏检（漏检帧 = 原脸闪回，是闪烁来源之一）
VIDEO_DET_THRESH = 0.4

def unload_model(model):
    if model is not None:
        # check if model has unload method
        # if "unload" in model:
        #     model.unload()
        # if "model_unload" in model:
        #     model.model_unload()
        del model
    return None

def unload_all_models():
    global FS_MODEL, CURRENT_FS_MODEL_PATH
    FS_MODEL = unload_model(FS_MODEL)
    ANALYSIS_MODELS["320"] = unload_model(ANALYSIS_MODELS["320"])
    ANALYSIS_MODELS["640"] = unload_model(ANALYSIS_MODELS["640"])

def get_current_faces_model():
    global SOURCE_FACES
    return SOURCE_FACES

def getAnalysisModel(det_size = (640, 640), det_thresh = 0.5):
    global ANALYSIS_MODELS
    ANALYSIS_MODEL = ANALYSIS_MODELS[str(det_size[0])]
    if ANALYSIS_MODEL is None:
        ANALYSIS_MODEL = insightface.app.FaceAnalysis(
            name="buffalo_l", providers=providers, root=insightface_path
        )
    ANALYSIS_MODEL.prepare(ctx_id=0, det_size=det_size, det_thresh=det_thresh)
    ANALYSIS_MODELS[str(det_size[0])] = ANALYSIS_MODEL
    return ANALYSIS_MODEL

# Кэш для типов моделей
_MODEL_TYPE_CACHE = {}

def _get_model_type(model_path: str):
    """Кэшированный определение типа модели для оптимизации"""
    if model_path not in _MODEL_TYPE_CACHE:
        model_filename = os.path.basename(model_path).lower()
        if "hyperswap" in model_filename:
            _MODEL_TYPE_CACHE[model_path] = "hyperswap"
        elif "reswapper" in model_filename:
            _MODEL_TYPE_CACHE[model_path] = "reswapper"
        else:
            _MODEL_TYPE_CACHE[model_path] = "insightface"
    return _MODEL_TYPE_CACHE[model_path]

def getFaceSwapModel(model_path: str):
    global FS_MODEL, CURRENT_FS_MODEL_PATH
    if FS_MODEL is None or CURRENT_FS_MODEL_PATH is None or CURRENT_FS_MODEL_PATH != model_path:
        CURRENT_FS_MODEL_PATH = model_path
        FS_MODEL = unload_model(FS_MODEL)

        model_type = _get_model_type(model_path)
        model_filename = os.path.basename(model_path)
        
        if model_type == "hyperswap":
            model_path = os.path.join(folder_paths.models_dir, "hyperswap", model_filename)
            FS_MODEL = ort.InferenceSession(model_path, providers=providers)
        elif model_type == "reswapper":
            model_path = os.path.join(folder_paths.models_dir, "reswapper", model_filename)
            FS_MODEL = insightface.model_zoo.get_model(model_path, providers=providers)
        else:
            FS_MODEL = insightface.model_zoo.get_model(model_path, providers=providers)

    return FS_MODEL


# Функция для получения 5 ключевых точек из объекта Face
def get_landmarks_5(face):
    if hasattr(face, 'landmark_5') and face.landmark_5 is not None:
        return face.landmark_5
    elif hasattr(face, 'kps') and face.kps is not None:
        return face.kps
    elif hasattr(face, 'landmark') and face.landmark is not None:
        if face.landmark.shape[0] >= 68:
            idxs = [36, 45, 30, 48, 54]
            return face.landmark[idxs]
    return None

# Функция для вычисления аффинного преобразования
def get_affine_transform(src_pts, dst_pts):
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    return M

# Кэш для масок разных размеров
_GRADIENT_MASK_CACHE = {}

# Создаём градиентную маску овальной формы без обрезки 
def create_gradient_mask(crop_size=256):
    # Используем кэш для избежания пересчёта маски одного размера
    if crop_size in _GRADIENT_MASK_CACHE:
        return _GRADIENT_MASK_CACHE[crop_size]
    
    # 1. Создаём пустую маску (все пиксели = 0)
    mask = np.zeros((crop_size, crop_size), dtype=np.float32)
    
    # 2. Определяем центр и размеры эллипса
    center = (crop_size // 2, crop_size // 2)
    axes = (int(crop_size * 0.35), int(crop_size * 0.4))
    
    # 3. Рисуем эллипс (заполняем белым цветом, значение=1.0)
    cv2.ellipse(
        mask,          # Массив для рисования
        center,        # Центр эллипса
        axes,          # Полуоси (ширина, высота)
        angle=0,       # Угол поворота
        startAngle=0,  # Начальный угол дуги
        endAngle=360,  # Конечный угол дуги (360 = полный эллипс)
        color=1.0,     # Значение для заполнения (белый = 1.0)
        thickness=-1   # -1 = заполнить всю область эллипса   
    )
    
    # 4. Применяем размытие для плавных краёв
    blur_ksize = 15  # Нечётное число, чтобы ядро было симметричным
    mask = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)
    
    # 5. Ограничим значения в диапазоне [0, 1]
    mask = np.clip(mask, 0, 1)
    
    # Сохраняем в кэш
    _GRADIENT_MASK_CACHE[crop_size] = mask
    
    return mask

def paste_back(target_img, swapped_face, M, crop_size=256):
    
    # 1. Создание мягкой маски (Эрозия + Размытие)
    mask = create_gradient_mask(crop_size)

    # Преобразуем в трехканальную маску
    mask_3c = np.stack([mask] * 3, axis=2)

    # 2. Получаем размеры целевого изображения
    h, w = target_img.shape[:2]

    # 3. Обратное преобразование (WARP_INVERSE_MAP) для лица И маски
    # Для лица (INTER_LANCZOS4 — высококачественная интерполяция)
    inv_face = cv2.warpAffine(
        swapped_face.astype(np.float32),
        M,
        (w, h),
        flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_TRANSPARENT
    )

    # Для маски (INTER_CUBIC — плавные границы)
    inv_mask = cv2.warpAffine(
        mask_3c,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_TRANSPARENT
    )

    # 4. Ограничение значений маски [0, 1]
    inv_mask = np.clip(inv_mask, 0, 1)

    # 5. Дополнительное размытие для устранения артефактов
    inv_mask = cv2.GaussianBlur(inv_mask, (3, 3), 0)

    # 6. Плавное наложение
    target_img_float = target_img.astype(np.float32)
    inv_face_float = inv_face.astype(np.float32)
    result = target_img_float * (1.0 - inv_mask) + inv_face_float * inv_mask

    # 7. Ограничение результата [0, 255]
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result

def visualize_points(img, points, color=(0, 255, 0)):
    img = img.copy()
    for p in points:
        cv2.circle(img, tuple(p.astype(int)), 3, color, -1)

# Итоговая функция run_hyperswap с аффинным преобразованием
def run_hyperswap(session, source_face, target_face, target_img):
    # 1. Подготовка эмбеддинга
    source_embedding = source_face.normed_embedding.reshape(1, -1).astype(np.float32)

    # 2. Получаем 5 точек target
    target_landmarks_5 = get_landmarks_5(target_face)
    visualize_points(target_img, target_landmarks_5, (0, 255, 0))  # Зеленые точки
    
    if target_landmarks_5 is None:
        return None, None

    # 3. Определение эталонных точек для выравнивания 256x256 (FFHQ Alignment)
    std_landmarks_256 = np.array([
        [ 84.87, 105.94],  # Левый глаз
        [171.13, 105.94],  # Правый глаз
        [128.00, 146.66],  # Кончик носа
        [ 96.95, 188.64],  # Левый уголок рта
        [159.05, 188.64]   # Правый уголок рта
    ], dtype=np.float32)

    # Вычисляем аффинную матрицу
    M = get_affine_transform(target_landmarks_5.astype(np.float32), std_landmarks_256)
    
    # Применяем аффинное преобразование с новой матрицей M
    crop = cv2.warpAffine(target_img, M, (256, 256), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    # 4. Преобразуем crop для модели
    crop_input = crop[:, :, ::-1].astype(np.float32) / 255.0  # RGB -> [0,1]
    crop_input = (crop_input - 0.5) / 0.5  # Нормализация
    crop_input = crop_input.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    # 5. Инференс
    try:
        output = session.run(None, {'source': source_embedding, 'target': crop_input})[0][0]
    except:
        return target_img

    # 6. Обратная нормализация
    output = (output * 0.5 + 0.5) * 255.0  # [-1..1] -> [0..255]
    output = np.clip(output, 0, 255).astype(np.uint8)
    output = output.transpose(1, 2, 0)  # CHW -> HWC
    output = output[:, :, ::-1]  # BGR -> RGB
    
    return output, M # Возвращаем лицо (256x256) и матрицу M


def sort_by_order(face, order: str):
    if order == "left-right":
        return sorted(face, key=lambda x: x.bbox[0])
    if order == "right-left":
        return sorted(face, key=lambda x: x.bbox[0], reverse = True)
    if order == "top-bottom":
        return sorted(face, key=lambda x: x.bbox[1])
    if order == "bottom-top":
        return sorted(face, key=lambda x: x.bbox[1], reverse = True)
    if order == "small-large":
        return sorted(face, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    # by default "large-small":
    return sorted(face, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse = True)

def get_face_gender(
    face,
    face_index,
    gender_condition,
    operated: str,
    order: str,
):
    filtered_faces = [
        f for f in face
        if (gender_condition == 0) or
        (gender_condition == 1 and f.sex == "F") or
        (gender_condition == 2 and f.sex == "M")
    ]

    gender = "Female" if gender_condition == 1 else "Male" if gender_condition == 0 else ""

    if len(filtered_faces) == 0:
        if gender_condition != 0:
            logger.status(f"No faces found for -{gender}-")
        return None, 0, None  # treat as "wrong gender" to skip

    faces_sorted = sort_by_order(filtered_faces, order)

    if face_index >= len(faces_sorted):
        logger.info("Requested face index (%s) is out of bounds (max available index is %s)", face_index, len(faces_sorted))
        return None, 0, None

    face_selected = faces_sorted[face_index]

    logger.info("%s Face %s: Detected Gender -%s-", operated, face_index, face_selected.sex)

    expected_gender = "F" if gender_condition == 1 else "M"
    if gender_condition != 0 and face_selected.sex != expected_gender:
        logger.info(f"{operated} Face {face_index}: WRONG gender ({face_selected.sex})")
        return face_selected, 1, face_index  # <-- есть, но не тот пол

    return face_selected, 0, face_index

def half_det_size(det_size):
    logger.status("Trying to halve 'det_size' parameter")
    return (det_size[0] // 2, det_size[1] // 2)

def analyze_faces(img_data: np.ndarray, det_size=(640, 640), det_thresh: float = 0.5):
    face_analyser = getAnalysisModel(det_size, det_thresh)

    faces = []
    try:
        faces = face_analyser.get(img_data)
    except:
        logger.error("No faces found")

    # Try halving det_size if no faces are found
    if len(faces) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = half_det_size(det_size)
        return analyze_faces(img_data, det_size_half, det_thresh)

    return faces

def get_face_single(img_data: np.ndarray, face, face_index=0, det_size=(640, 640), gender_source=0, gender_target=0, order="large-small"):

    buffalo_path = os.path.join(insightface_models_path, "buffalo_l.zip")
    if os.path.exists(buffalo_path):
        os.remove(buffalo_path)

    if gender_source != 0:
        if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
            det_size_half = half_det_size(det_size)
            return get_face_single(img_data, analyze_faces(img_data, det_size_half), face_index, det_size_half, gender_source, gender_target, order)
        return get_face_gender(face,face_index,gender_source,"Source", order)

    if gender_target != 0:
        if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
            det_size_half = half_det_size(det_size)
            return get_face_single(img_data, analyze_faces(img_data, det_size_half), face_index, det_size_half, gender_source, gender_target, order)
        return get_face_gender(face,face_index,gender_target,"Target", order)
    
    if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = half_det_size(det_size)
        return get_face_single(img_data, analyze_faces(img_data, det_size_half), face_index, det_size_half, gender_source, gender_target, order)

    try:
        faces_sorted = sort_by_order(face, order)
        return faces_sorted[face_index], 0, face_index
        # return sorted(face, key=lambda x: x.bbox[0])[face_index], 0
    except IndexError:
        return None, 0, None


def _bbox_iou(a, b) -> float:
    """计算两个 bbox [x1,y1,x2,y2] 的 IoU"""
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    iw = max(0.0, xx2 - xx1)
    ih = max(0.0, yy2 - yy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _get_face_embedding(face):
    """获取人脸嵌入向量（优先归一化嵌入），用于外观相似度匹配"""
    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        emb = getattr(face, "embedding", None)
        if emb is not None:
            norm = float(np.linalg.norm(emb))
            emb = emb / norm if norm > 1e-6 else None
    return emb


def track_face_across_frames(frames_faces, order, slot=0, taken=None):
    """
    时序人脸跟踪：在视频帧序列中为第 slot 号人脸槽位锁定同一个人。

    背景：逐帧独立按 faces_order 选脸时，主角转头（框变小）或运动模糊
    会导致下一帧跳选到背景中的人脸，或漏检导致原脸闪回，造成输出闪烁。

    策略：
      - 第一张有效帧按 order 排序取第 slot 个作为跟踪起点；
      - 后续帧在候选中按 IoU + 中心距离 + 面积相似度 + 嵌入相似度
        综合打分，选择与上一帧锁定人脸最匹配的；
      - 空间上完全不符（无重叠、距离过远、外观也不像）的候选直接排除，
        宁可该帧不换脸也不换错人；
      - 连续多帧丢失后重置跟踪，允许重新按 order 选脸。

    Args:
        frames_faces: list，每帧一个 face 列表
        order: faces_order 排序策略
        slot: 槽位号（在首帧排序中的名次）
        taken: 可选，list（每帧一个 set，存 id(face)），已被其他槽位占用的人脸

    Returns:
        list: 每帧对应的人脸（未匹配到时为 None）
    """
    n_frames = len(frames_faces)
    selected = [None] * n_frames
    prev_bbox = None
    prev_area = 0.0
    prev_emb = None
    miss_count = 0
    RESET_AFTER_MISSES = 15  # 连续丢失约半秒（30fps）后重置跟踪

    for i in range(n_frames):
        faces_all = list(frames_faces[i] or [])
        if len(faces_all) == 0:
            miss_count += 1
            if prev_bbox is not None and miss_count > RESET_AFTER_MISSES:
                prev_bbox = None
                prev_emb = None
            continue

        # 排序一次：初始选择用原始名次（保证多槽位各取各的脸），
        # 匹配阶段剔除已被其他槽位占用的人脸
        # 注：taken 存 id(face)——insightface Face 定义了 __eq__ 但不可哈希
        ordered = sort_by_order(faces_all, order)
        if taken is not None:
            taken_ids = taken[i]
            candidates = [f for f in ordered if id(f) not in taken_ids]
        else:
            candidates = ordered

        if prev_bbox is None:
            if slot < len(ordered):
                f = ordered[slot]
                if taken is None or id(f) not in taken[i]:
                    selected[i] = f
                    bbox = np.asarray(f.bbox, dtype=np.float64)
                    prev_bbox = bbox
                    prev_area = max(1e-6, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                    prev_emb = _get_face_embedding(f)
                    miss_count = 0
            continue

        if len(candidates) == 0:
            miss_count += 1
            continue

        diag = max(np.hypot(prev_bbox[2] - prev_bbox[0], prev_bbox[3] - prev_bbox[1]), 1e-6)
        pc = np.array([(prev_bbox[0] + prev_bbox[2]) / 2.0, (prev_bbox[1] + prev_bbox[3]) / 2.0])

        best_f = None
        best_score = 0.0
        for f in candidates:
            bbox = np.asarray(f.bbox, dtype=np.float64)
            iou = _bbox_iou(prev_bbox, bbox)
            cc = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
            dist = float(np.linalg.norm(cc - pc))
            dist_score = max(0.0, 1.0 - dist / (diag * 2.0))
            area = max(1e-6, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            size_score = max(0.0, 1.0 - abs(np.log(area / prev_area)))
            emb_sim = 0.0
            emb = _get_face_embedding(f)
            if prev_emb is not None and emb is not None:
                emb_sim = max(0.0, float(np.dot(prev_emb, emb)))
            # 空间/外观合理性门槛：与上一帧位置完全不符且外观也不像时直接排除
            if iou <= 0.01 and dist > diag * 2.5 and (prev_emb is None or emb_sim < 0.3):
                continue
            score = 1.5 * iou + 1.0 * dist_score + 0.5 * size_score + 0.8 * emb_sim
            if score > best_score:
                best_score = score
                best_f = f

        if best_f is not None and best_score >= 0.8:
            selected[i] = best_f
            bbox = np.asarray(best_f.bbox, dtype=np.float64)
            prev_bbox = bbox
            prev_area = max(1e-6, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            prev_emb = _get_face_embedding(best_f)
            miss_count = 0
        else:
            miss_count += 1
            if miss_count > RESET_AFTER_MISSES:
                prev_bbox = None
                prev_emb = None

    return selected


def smooth_face_angles(face_angles, window_size=5, alpha=0.4):
    """
    对面部角度序列进行平滑处理，减少检测抖动。
    缺失帧（None，未检测到脸）用相邻有效帧线性插值填充，
    避免以 0.0 参与平滑导致跳过/混合判断失真。

    Args:
        face_angles: 原始角度列表（元素为 float 或 None）
        window_size: 中值滤波窗口大小
        alpha: EWMA平滑系数，越小越平滑

    Returns:
        smoothed_angles: 平滑后的角度列表
    """
    import numpy as np

    if len(face_angles) <= 1:
        return [0.0 if a is None else float(a) for a in face_angles]

    # 第零步：填充缺失帧（未检测到脸）
    valid_idx = [i for i, a in enumerate(face_angles) if a is not None]
    if len(valid_idx) == 0:
        return [0.0 for _ in face_angles]

    filled = []
    for i, a in enumerate(face_angles):
        if a is not None:
            filled.append(float(a))
            continue
        prev_candidates = [j for j in valid_idx if j < i]
        next_candidates = [j for j in valid_idx if j > i]
        prev_i = prev_candidates[-1] if prev_candidates else None
        next_i = next_candidates[0] if next_candidates else None
        if prev_i is not None and next_i is not None:
            t = (i - prev_i) / float(next_i - prev_i)
            filled.append(face_angles[prev_i] * (1.0 - t) + face_angles[next_i] * t)
        elif prev_i is not None:
            filled.append(float(face_angles[prev_i]))
        else:
            filled.append(float(face_angles[next_i]))

    angles = np.array(filled)

    # 第一步：中值滤波消除异常值
    median_filtered = []
    half_window = window_size // 2

    for i in range(len(angles)):
        start = max(0, i - half_window)
        end = min(len(angles), i + half_window + 1)
        window_values = angles[start:end]
        median_filtered.append(np.median(window_values))

    # 第二步：指数加权移动平均（EWMA）进一步平滑
    smoothed = []
    smoothed.append(median_filtered[0])

    for i in range(1, len(median_filtered)):
        ewma = alpha * median_filtered[i] + (1 - alpha) * smoothed[-1]
        smoothed.append(ewma)

    return smoothed


def calculate_face_direction(face):
    """计算面部朝向角度（综合左右和上下偏转）"""
    import numpy as np
    
    DEBUG_MODE = False

    # 获取面部关键点
    kps = getattr(face, 'landmark_5', None)
    if kps is None:
        kps = getattr(face, 'kps', None)
    if kps is None:
        kps = getattr(face, 'landmark', None)
        if kps is not None and len(kps) >= 5:
            # 如果是68点，取前5个关键点位
            kps = kps[:5]

    if kps is None or len(kps) < 5:
        if DEBUG_MODE:
            print("[DEBUG] No valid face landmarks found")
        return 0.0
    
    # 打印关键点信息
    if DEBUG_MODE:
        print(f"[DEBUG] Face landmarks: {kps}")

    # 关键点索引：0=左眼，1=右眼，2=鼻子，3=左嘴角，4=右嘴角
    left_eye = np.array(kps[0])
    right_eye = np.array(kps[1])
    nose = np.array(kps[2])
    left_mouth = np.array(kps[3])
    right_mouth = np.array(kps[4])

    # 计算两眼之间的向量
    eye_vector = right_eye - left_eye
    # 计算鼻子到两眼中点的向量
    eye_midpoint = (left_eye + right_eye) / 2
    nose_vector = nose - eye_midpoint

    # 计算面部宽度和高度
    face_width = np.linalg.norm(eye_vector)
    face_height = np.linalg.norm(nose_vector)

    # 计算面部中心点
    face_center = (left_eye + right_eye + nose + left_mouth + right_mouth) / 5

    # 计算面部边界框
    face_points = np.array([left_eye, right_eye, nose, left_mouth, right_mouth])
    min_x = np.min(face_points[:, 0])
    max_x = np.max(face_points[:, 0])
    min_y = np.min(face_points[:, 1])
    max_y = np.max(face_points[:, 1])
    face_width_actual = max_x - min_x
    face_height_actual = max_y - min_y

    # ============ 计算左右偏转角度 ============
    # 计算左右面部的可见程度
    left_face_points = [left_eye, left_mouth]
    right_face_points = [right_eye, right_mouth]

    # 计算左侧面部点到中心的平均距离
    left_distances = [np.linalg.norm(p - face_center) for p in left_face_points]
    avg_left_distance = np.mean(left_distances)

    # 计算右侧面部点到中心的平均距离
    right_distances = [np.linalg.norm(p - face_center) for p in right_face_points]
    avg_right_distance = np.mean(right_distances)

    # 计算面部方向：基于左右面部可见程度
    horizontal_visibility_ratio = (avg_right_distance - avg_left_distance) / (max(avg_left_distance, avg_right_distance) + 1e-6)

    # 计算面部的宽高比，用于判断正面还是侧面
    width_height_ratio = face_width / (face_height + 1e-6)

    # 打印宽高比信息
    if DEBUG_MODE:
        print(f"[DEBUG] Width/height ratio: {width_height_ratio:.2f}")
    
    # 方法1：基于宽高比的角度计算（主要因素）
    if width_height_ratio > 1.5:
        # 正面
        angle_from_ratio = 0.0
    elif width_height_ratio < 0.9:
        # 侧脸
        angle_from_ratio = 85.0
    else:
        # 中间状态
        angle_from_ratio = (1.5 - width_height_ratio) / (1.5 - 0.9) * 85.0
    
    if DEBUG_MODE:
        print(f"[DEBUG] Angle from ratio: {angle_from_ratio:.2f}")

    # 方法2：基于可见度比例的角度增强
    visibility_strength = min(abs(horizontal_visibility_ratio) * 3.0, 1.0)
    angle_from_visibility = 85.0 * visibility_strength
    
    if DEBUG_MODE:
        print(f"[DEBUG] Horizontal visibility ratio: {horizontal_visibility_ratio:.2f}")
        print(f"[DEBUG] Angle from visibility: {angle_from_visibility:.2f}")

    # 综合两种方法，偏向于较大的角度
    horizontal_base_angle = max(angle_from_ratio, angle_from_visibility)

    # 强制增强：对于明显的侧脸，确保角度足够大
    if width_height_ratio < 1.1 or abs(horizontal_visibility_ratio) > 0.3:
        horizontal_base_angle = max(horizontal_base_angle, 75.0)

    # 计算左右偏转角度
    if horizontal_base_angle < 5.0 and abs(horizontal_visibility_ratio) < 0.1:
        # 接近正面
        horizontal_angle = 0.0
    else:
        # 根据可见度比例确定方向和角度大小
        if horizontal_visibility_ratio > 0:
            # 右脸更多（向左转）
            horizontal_angle = horizontal_base_angle
        elif horizontal_visibility_ratio < 0:
            # 左脸更多（向右转）
            horizontal_angle = -horizontal_base_angle
        else:
            # 左右脸相当
            horizontal_angle = 0.0

    # ============ 计算上下偏转角度 ============
    # 计算鼻子相对于两眼的位置（用于判断抬头/低头）
    mouth_midpoint = (left_mouth + right_mouth) / 2

    # 方法1：基于鼻子-眼睛与鼻子-嘴巴的垂直距离比例
    eye_nose_distance = np.linalg.norm(nose - eye_midpoint)
    nose_mouth_distance = np.linalg.norm(mouth_midpoint - nose)
    vertical_ratio = eye_nose_distance / (nose_mouth_distance + 1e-6)
    
    if DEBUG_MODE:
        print(f"[DEBUG] Eye-nose distance: {eye_nose_distance:.2f}, Nose-mouth distance: {nose_mouth_distance:.2f}")
        print(f"[DEBUG] Vertical ratio: {vertical_ratio:.2f}")

    # 方法2：基于上下面部关键点到中心的距离
    upper_face_points = [left_eye, right_eye, nose]
    lower_face_points = [left_mouth, right_mouth]

    upper_distances = [np.linalg.norm(p - face_center) for p in upper_face_points]
    avg_upper_distance = np.mean(upper_distances)

    lower_distances = [np.linalg.norm(p - face_center) for p in lower_face_points]
    avg_lower_distance = np.mean(lower_distances)

    vertical_visibility_ratio = (avg_lower_distance - avg_upper_distance) / (max(avg_upper_distance, avg_lower_distance) + 1e-6)
    
    if DEBUG_MODE:
        print(f"[DEBUG] Avg upper distance: {avg_upper_distance:.2f}, Avg lower distance: {avg_lower_distance:.2f}")
        print(f"[DEBUG] Vertical visibility ratio: {vertical_visibility_ratio:.2f}")

    # 方法3：基于面部实际宽高比
    face_aspect_ratio = face_width_actual / (face_height_actual + 1e-6)
    
    if DEBUG_MODE:
        print(f"[DEBUG] Face aspect ratio: {face_aspect_ratio:.2f}")

    # 综合计算上下偏转角度
    # 正常情况下，眼睛到鼻子和鼻子到嘴巴的距离比例约为0.8-1.2
    # 进一步调整阈值为更宽松的范围
    if vertical_ratio < 0.15 or vertical_ratio > 6.0:
        # 明显的抬头或低头
        vertical_angle_from_ratio = 85.0
    elif 0.25 <= vertical_ratio <= 5.5:
        # 正常范围
        vertical_angle_from_ratio = 0.0
    else:
        # 中间状态
        if vertical_ratio < 0.25:
            # 抬头
            vertical_angle_from_ratio = (0.25 - vertical_ratio) / (0.25 - 0.15) * 85.0
        else:
            # 低头
            vertical_angle_from_ratio = (vertical_ratio - 5.5) / (6.0 - 5.5) * 85.0

    # 基于可见度比例的角度增强
    vertical_visibility_strength = min(abs(vertical_visibility_ratio) * 2.0, 1.0)
    vertical_angle_from_visibility = 85.0 * vertical_visibility_strength

    # 基于宽高比的角度增强（抬头时脸部变窄，低头时变宽）
    if face_aspect_ratio < 0.6:
        # 抬头
        vertical_angle_from_aspect = 85.0 * (1.0 - face_aspect_ratio / 0.6)
    elif face_aspect_ratio > 1.4:
        # 低头
        vertical_angle_from_aspect = 85.0 * (face_aspect_ratio - 1.4) / 0.6
    else:
        vertical_angle_from_aspect = 0.0

    # 综合三种方法计算上下偏转（进一步降低可见度和宽高比的权重）
    vertical_base_angle = max(vertical_angle_from_ratio, vertical_angle_from_visibility * 0.3, vertical_angle_from_aspect * 0.2)

    # 确定上下偏转方向
    if vertical_base_angle < 5.0 and abs(vertical_visibility_ratio) < 0.35 and 0.25 <= vertical_ratio <= 5.5:
        # 接近正面
        vertical_angle = 0.0
    else:
        # 根据可见度比例和距离比例确定方向
        if vertical_visibility_ratio > 0.4 or vertical_ratio < 0.3:
            # 下巴更多/抬头
            vertical_angle = vertical_base_angle
        elif vertical_visibility_ratio < -0.4 or vertical_ratio > 5.0:
            # 额头更多/低头
            vertical_angle = -vertical_base_angle
        else:
            vertical_angle = 0.0

    # ============ 综合左右和上下角度 ============
    # 综合左右和上下角度
    # 使用欧几里得距离综合两个维度的角度
    combined_angle = np.sqrt(horizontal_angle ** 2 + vertical_angle ** 2)
    
    if DEBUG_MODE:
        print(f"[DEBUG] Horizontal angle: {horizontal_angle:.2f}, Vertical angle: {vertical_angle:.2f}")
        print(f"[DEBUG] Combined angle: {combined_angle:.2f}")

    return combined_angle


def smooth_blend_values(face_angles, angle_threshold, **_):
    """
    由（已平滑的）逐帧角度计算 blend 权重（原图权重）。

    输入角度序列已经过 smooth_face_angles 平滑（中值+EWMA），
    因此这里只做单调曲线映射，**不再对权重做时序平滑**：
      - 旧版对权重再做中值+EWMA 会引入明显滞后（转折后好帧仍被混入原图），
        且旧曲线在 threshold+5° 处不连续（smoothstep 段升到 1 后线性段又从 0 开始），
        导致权重几乎到不了 1.0、软跳过失效、并产生鬼影；
      - 单调映射保持角度序列的平滑性，无滞后、无跳变。

    曲线：angle <= threshold -> 0（完全换脸）
          angle >= threshold + 10 -> 1（完全还原原图，软跳过换脸）
          中间 smoothstep 平滑过渡

    Args:
        face_angles: 各帧的（平滑后）面部角度列表
        angle_threshold: 角度阈值

    Returns:
        weights: 逐帧原图权重列表（0=换脸，1=原图）
    """
    fade_end = float(angle_threshold) + 10.0
    weights = []
    for angle in face_angles:
        a = abs(float(angle))
        if a <= angle_threshold:
            w = 0.0
        elif a >= fade_end:
            w = 1.0
        else:
            t = (a - angle_threshold) / (fade_end - angle_threshold)
            w = t * t * (3.0 - 2.0 * t)  # smoothstep
        weights.append(max(0.0, min(1.0, w)))
    return weights


def swap_face(
    source_img: Union[Image.Image, None],
    target_img: Image.Image,
    model: Union[str, None] = None,
    source_faces_index: List[int] = [0],
    faces_index: List[int] = [0],
    gender_source: int = 0,
    gender_target: int = 0,
    face_model: Union[Face, None] = None,
    faces_order: List = ["large-small", "large-small"],
    face_boost_enabled: bool = False,
    face_restore_model = None,
    face_restore_visibility: int = 1,
    codeformer_weight: float = 0.5,
    interpolation: str = "Bicubic",
    angle_threshold: float = 60.0,
):
    global SOURCE_FACES, SOURCE_IMAGE_HASH, TARGET_FACES, TARGET_IMAGE_HASH
    result_image = target_img
    bbox = []
    swapped_indexes = []

    if model is not None:

        if isinstance(source_img, str):  # source_img is a base64 string
            import base64, io
            if 'base64,' in source_img:  # check if the base64 string has a data URL scheme
                # split the base64 string to get the actual base64 encoded image data
                base64_data = source_img.split('base64,')[-1]
                # decode base64 string to bytes
                img_bytes = base64.b64decode(base64_data)
            else:
                # if no data URL scheme, just decode
                img_bytes = base64.b64decode(source_img)
            
            source_img = Image.open(io.BytesIO(img_bytes))
            
        target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)

        if source_img is not None:

            source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)

            source_image_md5hash = get_image_md5hash(source_img)

            if SOURCE_IMAGE_HASH is None:
                SOURCE_IMAGE_HASH = source_image_md5hash
                source_image_same = False
            else:
                source_image_same = True if SOURCE_IMAGE_HASH == source_image_md5hash else False
                if not source_image_same:
                    SOURCE_IMAGE_HASH = source_image_md5hash

            logger.info("Source Image MD5 Hash = %s", SOURCE_IMAGE_HASH)
            logger.info("Source Image the Same? %s", source_image_same)

            if SOURCE_FACES is None or not source_image_same:
                logger.status("Analyzing Source Image...")
                source_faces = analyze_faces(source_img)
                SOURCE_FACES = source_faces
            elif source_image_same:
                logger.status("Using Hashed Source Face(s) Model...")
                source_faces = SOURCE_FACES

        elif face_model is not None:

            source_faces_index = [0]
            logger.status("Using Loaded Source Face Model...")
            source_face_model = [face_model]
            source_faces = source_face_model

        else:
            logger.error("Cannot detect any Source")

        if source_faces is not None:

            target_image_md5hash = get_image_md5hash(target_img)

            if TARGET_IMAGE_HASH is None:
                TARGET_IMAGE_HASH = target_image_md5hash
                target_image_same = False
            else:
                target_image_same = True if TARGET_IMAGE_HASH == target_image_md5hash else False
                if not target_image_same:
                    TARGET_IMAGE_HASH = target_image_md5hash

            logger.info("Target Image MD5 Hash = %s", TARGET_IMAGE_HASH)
            logger.info("Target Image the Same? %s", target_image_same)
            
            if TARGET_FACES is None or not target_image_same:
                logger.status("Analyzing Target Image...")
                target_faces = analyze_faces(target_img)
                TARGET_FACES = target_faces
            elif target_image_same:
                logger.status("Using Hashed Target Face(s) Model...")
                target_faces = TARGET_FACES

            # No use in trying to swap faces if no faces are found, enhancement
            if len(target_faces) == 0:
                logger.status("Cannot detect any Target, skipping swapping...")
                return result_image, bbox, swapped_indexes

            if source_img is not None:
                # separated management of wrong_gender between source and target, enhancement
                source_face, src_wrong_gender, source_face_index = get_face_single(source_img, source_faces, face_index=source_faces_index[0], gender_source=gender_source, order=faces_order[1])
            else:
                # source_face = sorted(source_faces, key=lambda x: x.bbox[0])[source_faces_index[0]]
                source_face = sorted(source_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse = True)[source_faces_index[0]]
                src_wrong_gender = 0

            if len(source_faces_index) != 0 and len(source_faces_index) != 1 and len(source_faces_index) != len(faces_index):
                logger.status(f'Source Faces must have no entries (default=0), one entry, or same number of entries as target faces.')
            elif source_face is not None:
                result = target_img
                if "inswapper" in model:
                    model_path = os.path.join(insightface_path, model)
                elif "reswapper" in model:
                    model_path = os.path.join(reswapper_path, model)
                elif "hyperswap" in model:
                    model_path = os.path.join(hyperswap_path, model)
                
                face_swapper = getFaceSwapModel(model_path)

                source_face_idx = 0

                for face_num in faces_index:
                    # No use in trying to swap faces if no further faces are found, enhancement
                    if face_num >= len(target_faces):
                        logger.status("Checked all existing target faces, skipping swapping...")
                        break

                    if len(source_faces_index) > 1 and source_face_idx > 0:
                        source_face, src_wrong_gender, source_face_index = get_face_single(source_img, source_faces, face_index=source_faces_index[source_face_idx], gender_source=gender_source, order=faces_order[1])
                    source_face_idx += 1

                    if source_face is not None and src_wrong_gender == 0:
                        target_face, wrong_gender, target_face_index = get_face_single(target_img, target_faces, face_index=face_num, gender_target=gender_target, order=faces_order[0])
                        if target_face is not None and wrong_gender == 0:
                            # 检查面部方向
                            face_angle = calculate_face_direction(target_face)
                            if abs(face_angle) <= angle_threshold:
                                logger.status(f"Swapping...")
                                if "hyperswap" in model:
                                    swapped_face_256, M = run_hyperswap(face_swapper, source_face, target_face, result)
                                    if swapped_face_256 is not None:
                                        result = paste_back(result, swapped_face_256, M, crop_size=256)
                                elif face_boost_enabled:
                                    logger.status(f"Face Boost is enabled (inswapper/reswapper only)")
                                    bgr_fake, M = face_swapper.get(result, target_face, source_face, paste_back=False)
                                    bgr_fake, scale = restorer.get_restored_face(bgr_fake, face_restore_model, face_restore_visibility, codeformer_weight, interpolation)
                                    M *= scale
                                    result = swapper.in_swap(result, bgr_fake, M)
                                else:
                                    result = face_swapper.get(result, target_face, source_face)
                                bbox = [tuple(map(float, target_face.bbox))]
                                swapped_indexes = [target_face_index]
                            else:
                                # logger.status(f"Face direction angle {abs(face_angle):.2f}° exceeds threshold {angle_threshold}°, skipping swap")
                                pass
                        elif wrong_gender == 1:
                            wrong_gender = 0
                            logger.status("Wrong target gender detected")
                            continue
                        else:
                            logger.info(f"No target face found for {face_num}")
                    elif src_wrong_gender == 1:
                        src_wrong_gender = 0
                        logger.status("Wrong source gender detected")
                        continue
                    else:
                        logger.status(f"No source face found for face number {source_face_idx}.")

                result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

            else:
                logger.status("No source face(s) in the provided Index")
        else:
            logger.status("No source face(s) found")
    return result_image, bbox, swapped_indexes

def swap_face_many(
    source_img: Union[Image.Image, None],
    target_imgs: List[Image.Image],
    model: Union[str, None] = None,
    source_faces_index: List[int] = [0],
    faces_index: List[int] = [0],
    gender_source: int = 0,
    gender_target: int = 0,
    face_model: Union[Face, None] = None,
    faces_order: List = ["large-small", "large-small"],
    face_boost_enabled: bool = False,
    face_restore_model = None,
    face_restore_visibility: int = 1,
    codeformer_weight: float = 0.5,
    interpolation: str = "Bicubic",
    angle_threshold: float = 60.0,
):
    global SOURCE_FACES, SOURCE_IMAGE_HASH, TARGET_FACES, TARGET_IMAGE_HASH, TARGET_FACES_LIST, TARGET_IMAGE_LIST_HASH
    result_images = target_imgs
    bbox = []  # 每帧一个列表: [[bbox,...], [], ...]，供 restore_face 按帧精确匹配
    swapped_indexes = []
    face_angles = []
    smoothed_face_angles = []
    blend_weights = []

    if model is not None:

        if isinstance(source_img, str):  # source_img is a base64 string
            import base64, io
            if 'base64,' in source_img:  # check if the base64 string has a data URL scheme
                # split the base64 string to get the actual base64 encoded image data
                base64_data = source_img.split('base64,')[-1]
                # decode base64 string to bytes
                img_bytes = base64.b64decode(base64_data)
            else:
                # if no data URL scheme, just decode
                img_bytes = base64.b64decode(source_img)
            
            source_img = Image.open(io.BytesIO(img_bytes))
            
        target_imgs = [cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR) for target_img in target_imgs]

        if source_img is not None:

            source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)

            source_image_md5hash = get_image_md5hash(source_img)

            if SOURCE_IMAGE_HASH is None:
                SOURCE_IMAGE_HASH = source_image_md5hash
                source_image_same = False
            else:
                source_image_same = True if SOURCE_IMAGE_HASH == source_image_md5hash else False
                if not source_image_same:
                    SOURCE_IMAGE_HASH = source_image_md5hash

            logger.info("Source Image MD5 Hash = %s", SOURCE_IMAGE_HASH)
            logger.info("Source Image the Same? %s", source_image_same)

            if SOURCE_FACES is None or not source_image_same:
                logger.status("Analyzing Source Image...")
                source_faces = analyze_faces(source_img)
                SOURCE_FACES = source_faces
            elif source_image_same:
                logger.status("Using Hashed Source Face(s) Model...")
                source_faces = SOURCE_FACES

        elif face_model is not None:

            source_faces_index = [0]
            logger.status("Using Loaded Source Face Model...")
            source_face_model = [face_model]
            source_faces = source_face_model

        else:
            logger.error("Cannot detect any Source")

        if source_faces is not None:

            target_faces = []  # 每帧一个人脸列表（可能为空列表）
            pbar = progress_bar(len(target_imgs))

            if len(TARGET_IMAGE_LIST_HASH) > 0:
                logger.status(f"Using Hashed Target Face(s) Model...")
            else:
                logger.status(f"Analyzing Target Image...")

            for i, target_img in enumerate(target_imgs):
                if state.interrupted or model_management.processing_interrupted():
                    logger.status("Interrupted by User")
                    break

                target_image_md5hash = get_image_md5hash(target_img)
                if len(TARGET_IMAGE_LIST_HASH) == 0:
                    TARGET_IMAGE_LIST_HASH = [target_image_md5hash]
                    target_image_same = False
                elif len(TARGET_IMAGE_LIST_HASH) == i:
                    TARGET_IMAGE_LIST_HASH.append(target_image_md5hash)
                    target_image_same = False
                else:
                    target_image_same = True if TARGET_IMAGE_LIST_HASH[i] == target_image_md5hash else False
                    if not target_image_same:
                        TARGET_IMAGE_LIST_HASH[i] = target_image_md5hash

                logger.info("(Image %s) Target Image MD5 Hash = %s", i, TARGET_IMAGE_LIST_HASH[i])
                logger.info("(Image %s) Target Image the Same? %s", i, target_image_same)

                if len(TARGET_FACES_LIST) == 0:
                    # logger.status(f"Analyzing Target Image {i}...")
                    target_face = analyze_faces(target_img, det_thresh=VIDEO_DET_THRESH)
                    TARGET_FACES_LIST = [target_face]
                elif len(TARGET_FACES_LIST) == i and not target_image_same:
                    # logger.status(f"Analyzing Target Image {i}...")
                    target_face = analyze_faces(target_img, det_thresh=VIDEO_DET_THRESH)
                    TARGET_FACES_LIST.append(target_face)
                elif len(TARGET_FACES_LIST) != i and not target_image_same:
                    # logger.status(f"Analyzing Target Image {i}...")
                    target_face = analyze_faces(target_img, det_thresh=VIDEO_DET_THRESH)
                    TARGET_FACES_LIST[i] = target_face
                elif target_image_same:
                    # logger.status("(Image %s) Using Hashed Target Face(s) Model...", i)
                    target_face = TARGET_FACES_LIST[i]

                target_faces.append(target_face if target_face is not None else [])

                pbar.update(1)

            progress_bar_reset(pbar)

            # 若分析被中断，补齐每帧条目，保证后续按帧索引访问不越界
            while len(target_faces) < len(target_imgs):
                target_faces.append([])

            # No use in trying to swap faces if no faces are found, enhancement
            if not any(len(tf) > 0 for tf in target_faces):
                logger.status("Cannot detect any Target, skipping swapping...")
                return result_images, bbox, swapped_indexes, face_angles, blend_weights

            # 每帧一个 bbox 列表（与帧一一对应），供 restore_face 按帧精确匹配
            bbox = [[] for _ in target_imgs]

            # ---- 时序人脸跟踪：为每个目标槽位在全部帧中锁定同一个人 ----
            # 逐帧独立选脸会在主角转头/运动模糊时跳选到背景人脸或漏检，
            # 造成换脸/不换脸反复交替（闪烁）。
            n_frames = len(target_imgs)
            if gender_target != 0:
                wanted_sex = "F" if gender_target == 1 else "M"
                frames_for_track = [
                    [f for f in tf if getattr(f, "sex", None) == wanted_sex] for tf in target_faces
                ]
            else:
                frames_for_track = target_faces

            tracked_lists = []
            taken = [set() for _ in range(n_frames)]
            for slot, face_num in enumerate(faces_index):
                tracked = track_face_across_frames(frames_for_track, faces_order[0], slot=slot, taken=taken)
                for i, f in enumerate(tracked):
                    if f is not None:
                        taken[i].add(id(f))
                tracked_lists.append(tracked)

            # ---- 用主槽位的跟踪结果计算逐帧角度（缺失帧为 None，平滑时自动插值） ----
            primary_tracked = tracked_lists[0] if tracked_lists else [None] * n_frames
            face_angles = [calculate_face_direction(f) if f is not None else None for f in primary_tracked]

            # 对角度序列进行平滑处理，减少检测抖动
            smoothed_face_angles = smooth_face_angles(face_angles)
            logger.status(
                f"Face angles smoothed. Original: {[('n/a' if a is None else f'{a:.1f}') for a in face_angles[:10]]}... "
                f"-> Smoothed: {[f'{a:.1f}' for a in smoothed_face_angles[:10]]}..."
            )

            # ---- 预计算 blend 权重，并以此做"软跳过" ----
            # 旧逻辑：角度 > threshold 的帧直接不换脸（硬切换，输出闪烁）。
            # 新逻辑：只要权重未到 1.0（未完全还原原图）就执行换脸，
            #         过渡区间 [threshold, 75°] 由后处理按权重淡出，平滑无跳变。
            blend_weights = smooth_blend_values(smoothed_face_angles, angle_threshold)

            if source_img is not None:
                # separated management of wrong_gender between source and target, enhancement
                source_face, src_wrong_gender, source_face_index = get_face_single(source_img, source_faces, face_index=source_faces_index[0], gender_source=gender_source, order=faces_order[1])
            else:
                # source_face = sorted(source_faces, key=lambda x: x.bbox[0])[source_faces_index[0]]
                source_face = sorted(source_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse = True)[source_faces_index[0]]
                src_wrong_gender = 0

            if len(source_faces_index) != 0 and len(source_faces_index) != 1 and len(source_faces_index) != len(faces_index):
                logger.status(f'Source Faces must have no entries (default=0), one entry, or same number of entries as target faces.')
            elif source_face is not None:
                results = target_imgs
                if "inswapper" in model:
                    model_path = os.path.join(insightface_path, model)
                elif "reswapper" in model:
                    model_path = os.path.join(reswapper_path, model)
                elif "hyperswap" in model:
                    model_path = os.path.join(hyperswap_path, model)

                face_swapper = getFaceSwapModel(model_path)

                source_face_idx = 0

                pbar = progress_bar(len(target_imgs))

                logger.status(f"Swapping...")
                for slot, face_num in enumerate(faces_index):
                    if slot >= len(tracked_lists):
                        logger.status("Checked all existing target faces, skipping swapping...")
                        break

                    if len(source_faces_index) > 1 and source_face_idx > 0:
                        source_face, src_wrong_gender, source_face_index = get_face_single(source_img, source_faces, face_index=source_faces_index[source_face_idx], gender_source=gender_source, order=faces_order[1])
                    source_face_idx += 1

                    if source_face is not None and src_wrong_gender == 0:
                        # Reading results to make current face swap on a previous face result
                        # logger.status(f"Swapping...")
                        tracked_slot = tracked_lists[slot]
                        for i, target_img in enumerate(results):
                            target_face_single = tracked_slot[i] if i < len(tracked_slot) else None
                            # 软跳过：blend 权重达到 1.0（完全还原原图）才跳过，
                            # 过渡区间照常换脸、由后处理淡出，避免相邻帧硬切换闪烁
                            if target_face_single is not None and blend_weights[i] < 0.999:
                                result = target_img
                                if "hyperswap" in model:
                                    swapped_face_256, M = run_hyperswap(face_swapper, source_face, target_face_single, result)
                                    if swapped_face_256 is not None:
                                        result = paste_back(result, swapped_face_256, M, crop_size=256)
                                elif face_boost_enabled:
                                    logger.status(f"Face Boost is enabled (inswapper/reswapper only)")
                                    bgr_fake, M = face_swapper.get(target_img, target_face_single, source_face, paste_back=False)
                                    bgr_fake, scale = restorer.get_restored_face(bgr_fake, face_restore_model, face_restore_visibility, codeformer_weight, interpolation)
                                    M *= scale
                                    result = swapper.in_swap(target_img, bgr_fake, M)
                                else:
                                    result = face_swapper.get(target_img, target_face_single, source_face)
                                results[i] = result
                                bbox[i].append(tuple(map(float, target_face_single.bbox)))
                                swapped_indexes.append(face_num)
                                pbar.update(1)
                            else:
                                pbar.update(1)
                    elif src_wrong_gender == 1:
                        src_wrong_gender = 0
                        logger.status("Wrong source gender detected")
                        continue
                    else:
                        logger.status(f"No source face found for face number {source_face_idx}.")

                progress_bar_reset(pbar)

                result_images = [Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)) for result in results]

            else:
                logger.status("No source face(s) in the provided Index")
        else:
            logger.status("No source face(s) found")
    return result_images, bbox, swapped_indexes, smoothed_face_angles, blend_weights
