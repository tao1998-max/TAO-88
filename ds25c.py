import cv2  # 导入OpenCV库，用于图像处理
import numpy as np  # 导入NumPy库，用于数值计算
import math  # 导入数学库，用于基本数学运算

# 全局变量用于存储相机焦距和对焦值
focal_length = None  # 初始化焦距为None，需通过标定获取
current_focus = 170  # 初始化当前对焦值为0

def set_camera_focus(cap, focus_value):
    """设置相机对焦值"""
    try:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 关闭自动对焦
        cap.set(cv2.CAP_PROP_FOCUS, focus_value)  # 设置手动对焦值
        return True
    except Exception as e:
        print(f"设置对焦失败: {e}")
        return False

def calibrate_focal_length(frame, known_distance_mm, reference_size_pixels):
    """标定相机焦距，使用已知距离和A4纸的像素尺寸"""
    a4_physical_width_mm = 210  # A4纸外框宽度210mm
    if reference_size_pixels == 0:
        return None
    focal = (reference_size_pixels * known_distance_mm) / a4_physical_width_mm
    return focal

def calculate_distance(pixel_width, focal_length, physical_width_mm=210):
    """根据像素宽度和焦距计算到A4纸外框中心点的距离（毫米）"""
    if focal_length is None or pixel_width == 0:
        return None
    distance = (physical_width_mm * focal_length) / pixel_width
    return distance

def calculate_physical_size(pixel_size, reference_size_pixels):
    """将像素尺寸转换为物理尺寸（毫米）"""
    if reference_size_pixels == 0:
        return 0
    return (pixel_size / reference_size_pixels) * 170  # 内框参考宽度170mm

def order_points(pts):
    """对矩形四个顶点进行排序：左上、右上、右下、左下"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """应用透视变换校正图像"""
    if len(pts) != 4:
        return image, np.eye(3, 3, dtype=np.float32)
    cv2.circle(image, tuple(pts[0]), 5, (0, 0, 255), -1)  # 红色
    cv2.circle(image, tuple(pts[1]), 5, (0, 255, 0), -1)  # 绿色
    cv2.circle(image, tuple(pts[2]), 5, (255, 0, 0), -1)  # 蓝色
    cv2.circle(image, tuple(pts[3]), 5, (0, 255, 255), -1)  # 黄色
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2 + ((br[1] - bl[1]) ** 2)))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2 + ((tr[1] - tl[1]) ** 2)))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2 + ((tr[1] - br[1]) ** 2)))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2 + ((tl[1] - bl[1]) ** 2)))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped, M

def detect_a4_border(frame):
    """检测可能的A4纸边框，返回最大矩形轮廓"""
    if frame is None or frame.size == 0:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    border_contour = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area:
                hull = cv2.convexHull(approx)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                if solidity > 0.9 and (0.7 < aspect_ratio < 1.3):
                    max_area = area
                    border_contour = approx
    return border_contour

def detect_marker(frame, inner_border):
    """检测底边中点标记点"""
    if frame is None or frame.size == 0 or inner_border is None:
        return None
    try:
        bottom_center = ((inner_border[2][0] + inner_border[3][0]) // 2,
                        (inner_border[2][1] + inner_border[3][1]) // 2)
        roi_size = 100
        x = max(0, bottom_center[0] - roi_size//2)
        y = max(0, bottom_center[1] - roi_size//2)
        roi = frame[y:y+roi_size, x:x+roi_size]
        if roi.size == 0:
            return None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
                                 param1=50, param2=30, minRadius=5, maxRadius=30)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            return (circles[0, 0][0] + x, circles[0, 0][1] + y), circles[0, 0][2]
    except:
        pass
    return None

def detect_shape(roi):
    """改进的形状检测函数"""
    shape_info = {
        "name": "Unknown",  # 使用英文名称以匹配画面显示
        "size_mm": 0,
        "contour": None,
        "approx": None
    }
    if roi is None or roi.size == 0:
        return shape_info
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
            num_sides = len(approx)
            if num_sides == 3:
                shape_info.update({
                    "name": "Triangle",
                    "size_px": peri / 3,
                    "contour": contour,
                    "approx": approx
                })
                break
            elif num_sides == 4:
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                if solidity > 0.85 and (0.7 < aspect_ratio < 1.3):
                    shape_info.update({
                        "name": "Square" if 0.9 < aspect_ratio < 1.1 else "Rectangle",
                        "size_px": (w + h) / 2,
                        "contour": contour,
                        "approx": approx
                    })
                    break
            else:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                area_ratio = area / (math.pi * radius**2)
                if area_ratio > 0.7:
                    shape_info.update({
                        "name": "Circle",
                        "size_px": 2 * radius,
                        "contour": contour,
                        "approx": approx
                    })
                    break
    except Exception as e:
        print(f"形状检测错误: {e}")
    return shape_info

# 主程序
cap = cv2.VideoCapture(1)  # 打开默认摄像头
if not cap.isOpened():
    raise ValueError("无法打开摄像头")

# 初始化相机对焦
set_camera_focus(cap, current_focus)

reference_size_pixels = None
shape_size_mm = 0
shape_name = "Unknown"
distance_mm = None
calibration_mode = False

print("按 'c' 进入标定模式，按 'u' 增加对焦，按 'd' 减少对焦，按 'q' 退出程序")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("无法获取视频帧")
        break

    border_contour = detect_a4_border(frame)
    
    if border_contour is not None:
        warped, M = four_point_transform(frame, border_contour.reshape(4, 2))
        border_ratio = 20/210
        try:
            warped, M = four_point_transform(frame, border_contour.reshape(4, 2))
            h, w = warped.shape[:2]
            border_width_px = int(border_ratio * w)
            inner_border = np.array([
                [border_width_px, border_width_px],
                [w - border_width_px, border_width_px],
                [w - border_width_px, h - border_width_px],
                [border_width_px, h - border_width_px]
            ], dtype=np.int32)
            # 使用外框宽度计算距离
            reference_size_pixels = w  # 外框像素宽度
            # 计算绿色外框中心点
            border_center = np.mean(border_contour.reshape(4, 2), axis=0).astype(np.int32)
            # 绘制外框中心点（绿色圆点）
            cv2.circle(frame, tuple(border_center), 5, (0, 255, 0), -1)
            cv2.putText(frame, "Frame Center", (border_center[0] + 10, border_center[1]), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
            
            # 计算A4纸外框中心点的距离
            if focal_length is not None:
                distance_mm = calculate_distance(reference_size_pixels, focal_length)
            
            # 标定模式
            if calibration_mode:
                known_distance = float(input("请输入A4纸到相机的已知距离（毫米）："))
                focal_length = calibrate_focal_length(frame, known_distance, reference_size_pixels)
                print(f"焦距标定完成：{focal_length:.1f} 像素")
                calibration_mode = False
            
            # 检测并绘制底边中点标记
            marker_info = detect_marker(warped, inner_border)
            if marker_info is not None:
                marker_center, marker_radius = marker_info
                marker_center = np.array([[marker_center]], dtype=np.float32)
                marker_center_original = cv2.perspectiveTransform(marker_center, np.linalg.inv(M))
                marker_center_original = tuple(np.int32(marker_center_original[0][0]))
                # 绘制标记点（紫色圆形）
                cv2.circle(frame, marker_center_original, int(marker_radius), (255, 0, 255), 2)
                cv2.putText(frame, "Center Marker", (marker_center_original[0] + 10, marker_center_original[1]), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 0, 255), 2)
            
            roi_x1 = max(0, inner_border[0][0])
            roi_y1 = max(0, inner_border[0][1])
            roi_x2 = min(w, inner_border[1][0])
            roi_y2 = min(h, inner_border[2][1])
            roi = warped[roi_y1:roi_y2, roi_x1:roi_x2]
            
            if roi.size > 0:
                shape_info = detect_shape(roi)
                if shape_info["name"] != "Unknown":
                    shape_name = shape_info["name"]
                    shape_size_mm = calculate_physical_size(
                        shape_info["size_px"], 
                        inner_border[1][0] - inner_border[0][0]  # 使用内框宽度作为参考
                    )
                    contour_global = shape_info["contour"] + np.array([roi_x1, roi_y1])
                    contour_original = cv2.perspectiveTransform(
                        contour_global.reshape(-1, 1, 2).astype(np.float32), 
                        np.linalg.inv(M)
                    ).astype(np.int32)
                    cv2.drawContours(frame, [contour_original], -1, (0, 0, 255), 2)
                    if shape_name == "Circle":
                        text = f"Diameter: {shape_size_mm:.1f} mm"
                    else:
                        text = f"{shape_name}: {shape_size_mm:.1f} mm"
                    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            
            # 显示距离信息（英文）
            if distance_mm is not None:
                cv2.putText(frame, f"Distance: {distance_mm:.1f} mm", (50, 100), 
                           cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
            
            # 显示当前对焦值（英文）
            cv2.putText(frame, f"Focus: {current_focus}", (50, 150), 
                       cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2)
            
            cv2.drawContours(frame, [border_contour], -1, (0, 255, 0), 2)
            inner_border_original = cv2.perspectiveTransform(
                inner_border.reshape(-1, 1, 2).astype(np.float32),
                np.linalg.inv(M)
            ).astype(np.int32)
            cv2.drawContours(frame, [inner_border_original], -1, (255, 0, 0), 2)
        except Exception as e:
            print(f"处理错误: {e}")
    
    cv2.imshow('Real-time Shape Detection', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        calibration_mode = True
        print("进入标定模式，请确保A4纸清晰可见")
    elif key == ord('u'):
        current_focus = min(current_focus + 1, 255)  # 增加对焦值，最大255
        if set_camera_focus(cap, current_focus):
            print(f"对焦值增加到: {current_focus}")
    elif key == ord('d'):
        current_focus = max(current_focus - 1, 0)  # 减少对焦值，最小0
        if set_camera_focus(cap, current_focus):
            print(f"对焦值减少到: {current_focus}")

cap.release()
cv2.destroyAllWindows()

if shape_size_mm > 0:
    if shape_name == "Circle":
        print(f"检测到{shape_name}，直径为{shape_size_mm:.1f}毫米")
    else:
        print(f"检测到{shape_name}，边长为{shape_size_mm:.1f}毫米")
if distance_mm is not None:
    print(f"相机到A4纸外框中心点的距离：{distance_mm:.1f}毫米")