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

    RETURN_TYPES = ("IMAGE","FACE_MODEL","IMAGE","FLOAT","STRING")
    RETURN_NAMES = ("SWAPPED_IMAGE","FACE_MODEL","ORIGINAL_IMAGE","FACE_ANGLE","FACE_DIRECTION")
    FUNCTION = "execute"
    CATEGORY = "🌌 ReActor"

    def __init__(self):
        # 基本参数初始化
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

    def calculate_face_direction(self, input_image):
        """计算面部朝向角度"""
        import numpy as np
        from scripts.reactor_faceswap import analyze_faces

        # 将IMAGE张量转换为numpy数组
        def tensor_to_image(tensor):
            if len(tensor.shape) == 4:
                tensor = tensor[0]
            tensor = tensor * 255
            tensor = tensor.clamp(0, 255)
            return tensor.cpu().numpy().astype(np.uint8)

        # 处理图像
        face_img = tensor_to_image(input_image)

        # 使用ReActor的面部检测
        faces = analyze_faces(face_img)

        if not faces:
            return (0.0, "No face detected")

        # 获取第一个面部
        face = faces[0]

        # 获取面部关键点
        kps = face.get('kps', [])
        if not kps:
            # 尝试从其他属性获取关键点
            kps = getattr(face, 'landmark_5', None)
        if not kps:
            kps = getattr(face, 'landmark', None)
            if kps and len(kps) >= 5:
                # 如果是68点，取前5个关键点位
                kps = kps[:5]

        if not kps or len(kps) < 5:
            return (0.0, "Insufficient keypoints")

        # 关键点索引：0=左眼，1=右眼，2=鼻子，3=左嘴角，4=右嘴角
        left_eye = kps[0]
        right_eye = kps[1]
        nose = kps[2]

        # 计算两眼之间的向量
        eye_vector = np.array(right_eye) - np.array(left_eye)
        # 计算鼻子到两眼中点的向量
        eye_midpoint = (np.array(left_eye) + np.array(right_eye)) / 2
        nose_vector = np.array(nose) - eye_midpoint

        # 计算面部宽度和高度
        face_width = np.linalg.norm(eye_vector)
        face_height = np.linalg.norm(nose_vector)

        # 分析面部关键点，确定可见面部面积
        left_eye = np.array(kps[0])
        right_eye = np.array(kps[1])
        nose = np.array(kps[2])
        left_mouth = np.array(kps[3])
        right_mouth = np.array(kps[4])

        # 计算面部中心点
        face_center = (left_eye + right_eye + nose + left_mouth + right_mouth) / 5

        # 计算面部边界框
        face_points = np.array([left_eye, right_eye, nose, left_mouth, right_mouth])
        min_x = np.min(face_points[:, 0])
        max_x = np.max(face_points[:, 0])
        face_width_actual = max_x - min_x

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
        visibility_ratio = (avg_right_distance - avg_left_distance) / (max(avg_left_distance, avg_right_distance) + 1e-6)

        # 计算面部的宽高比，用于判断正面还是侧面
        width_height_ratio = face_width / (face_height + 1e-6)

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

        # 方法2：基于可见度比例的角度增强
        visibility_strength = min(abs(visibility_ratio) * 3.0, 1.0)
        angle_from_visibility = 85.0 * visibility_strength

        # 综合两种方法，偏向于较大的角度
        base_angle = max(angle_from_ratio, angle_from_visibility)

        # 强制增强：对于明显的侧脸，确保角度足够大
        if width_height_ratio < 1.1 or abs(visibility_ratio) > 0.3:
            base_angle = max(base_angle, 75.0)

        # 计算最终角度
        if base_angle < 5.0 and abs(visibility_ratio) < 0.1:
            # 接近正面
            direction_angle = 0.0
        else:
            # 根据可见度比例确定方向和角度大小
            if visibility_ratio > 0:
                # 右脸更多
                direction_angle = base_angle
            elif visibility_ratio < 0:
                # 左脸更多
                direction_angle = -base_angle
            else:
                # 左右脸相当
                direction_angle = 0.0

        # 确定方向描述
        if abs(direction_angle) < 10:
            direction = "Front"
        elif direction_angle > 45:
            direction = "Right Side"
        elif direction_angle < -45:
            direction = "Left Side"
        elif direction_angle > 0:
            direction = "Right Quarter"
        else:
            direction = "Left Quarter"

        return (direction_angle, direction)

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

        # 计算面部朝向
        face_angle, face_direction = self.calculate_face_direction(input_image)

        # 检查面部朝向是否符合阈值要求
        if enabled and abs(face_angle) <= angle_threshold:
            # 执行正常的人脸替换
            result = reactor.execute(
                self,enabled,input_image,swap_model,self.detect_gender_source,self.detect_gender_input,self.source_faces_index,self.input_faces_index,self.console_log_level,face_restore_model,face_restore_visibility,codeformer_weight,facedetection,source_image,face_model,self.faces_order, face_boost=face_boost
            )
            # 扩展返回值，添加面部角度和方向
            return (*result, face_angle, face_direction)
        else:
            # 不符合条件，直接返回原图
            if face_model is None:
                # 如果没有提供face_model，返回None
                return (input_image, None, input_image, face_angle, face_direction)
            else:
                return (input_image, face_model, input_image, face_angle, face_direction)