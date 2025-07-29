import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen

# 参数设置
d = 10.0  # 平面距离 z = d
X = 4.0   # 矩形宽
Y = 3.0   # 矩形高
spot_size = 0.2  # 激光光斑的直径（圆形）

# 计算步进电机角度的最大值
alpha_max = np.arctan(Y / (2 * d))  # YZ 平面角度（y 方向电机）
beta_max = np.arctan(X / (2 * d))   # XZ 平面角度（x 方向电机）

# 自定义笔画提取类
class PointPen(BasePen):
    def __init__(self):
        super().__init__(None)
        self.points = []
        self.current_point = None

    def _moveTo(self, pt):
        self.current_point = pt
        self.points.append(pt)

    def _lineTo(self, pt):
        self.current_point = pt
        self.points.append(pt)

    def _curveToOne(self, pt1, pt2, pt3):
        # 简化曲线为直线段
        steps = 10
        t = np.linspace(0, 1, steps)
        for ti in t:
            x = (1-ti)**3 * self.current_point[0] + 3*(1-ti)**2 * ti * pt1[0] + 3*(1-ti) * ti**2 * pt2[0] + ti**3 * pt3[0]
            y = (1-ti)**3 * self.current_point[1] + 3*(1-ti)**2 * ti * pt1[1] + 3*(1-ti) * ti**2 * pt2[1] + ti**3 * pt3[1]
            self.points.append((x, y))
        self.current_point = pt3

    def _closePath(self):
        pass

# 定义自定义路径（生成汉字路径，匀速移动）
def get_custom_path():
    # 字体文件路径和目标汉字
    font_path = 'MapleMono-CN-Bold.ttf' 
    char = '牛' 

    try:
        # 加载字体并提取汉字轮廓
        font = TTFont(font_path)
        unicode_char = ord(char)
        glyph_name = font.getBestCmap().get(unicode_char)
        if glyph_name is None:
            raise ValueError(f"Font does not support character: {char}")
        glyph = font['glyf'][glyph_name]
        glyf_table = font['glyf']  # 获取 glyf 表

        # 提取轮廓点
        pen = PointPen()
        glyph.draw(pen, glyf_table)  # 传递 glyf_table 参数
        points = np.array(pen.points, dtype=float)

        if len(points) == 0:
            raise ValueError("No points extracted from glyph")

        # 缩放到矩形区域
        min_x, max_x = points[:, 0].min(), points[:, 0].max()
        min_y, max_y = points[:, 1].min(), points[:, 1].max()
        if max_x > min_x and max_y > min_y:
            points[:, 0] = (points[:, 0] - min_x) / (max_x - min_x) * X * 0.8 - X/2
            points[:, 1] = (points[:, 1] - min_y) / (max_y - min_y) * Y * 0.8 - Y/2
        else:
            points[:, 0] = points[:, 0] - X/2
            points[:, 1] = points[:, 1] - Y/2

        # 计算弧长并按弧长均匀插值
        steps = 300
        if len(points) > 1:
            # 计算每段距离
            diffs = np.diff(points, axis=0)
            segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
            arc_lengths = np.cumsum(np.concatenate([[0], segment_lengths]))
            total_length = arc_lengths[-1]
            # 按弧长均匀采样
            s_new = np.linspace(0, total_length, steps)
            x = np.interp(s_new, arc_lengths, points[:, 0])
            y = np.interp(s_new, arc_lengths, points[:, 1])
            points = np.array(list(zip(x, y)))

        return points

    except Exception as e:
        print(f"Error loading font or glyph: {e}. Using fallback path (horizontal line).")
        # 后备路径（水平直线）
        steps = 300
        x = np.linspace(-X/2 * 0.8, X/2 * 0.8, steps)
        y = np.zeros_like(x)
        return np.array(list(zip(x, y)))

# 将平面坐标 (x, y) 转换为步进电机角度 (alpha, beta)
def xy_to_angles(x, y):
    alpha = np.arctan(y / d)  # YZ 平面投影角度（y 方向电机）
    beta = np.arctan(x / d)   # XZ 平面投影角度（x 方向电机）
    return alpha, beta

# 设置画布：包含 3D 和 2D 子图
fig = plt.figure(figsize=(12, 6))

# 3D 子图：显示激光束路径
ax_3d = fig.add_subplot(121, projection='3d')
ax_3d.set_xlim(-X/2 - 1, X/2 + 1)
ax_3d.set_ylim(-Y/2 - 1, Y/2 + 1)
ax_3d.set_zlim(0, d + 1)
ax_3d.set_xlabel('X')
ax_3d.set_ylabel('Y')
ax_3d.set_zlabel('Z')
ax_3d.set_title('3D Laser Beam Path')

# 绘制目标平面
x_plane = np.linspace(-X/2, X/2, 10)
y_plane = np.linspace(-Y/2, Y/2, 10)
X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
Z_plane = np.ones_like(X_plane) * d
ax_3d.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.2, color='blue')

# 绘制矩形边界
rect_x = [-X/2, X/2, X/2, -X/2, -X/2]
rect_y = [-Y/2, -Y/2, Y/2, Y/2, -Y/2]
rect_z = [d, d, d, d, d]
ax_3d.plot(rect_x, rect_y, rect_z, color='blue', linewidth=2)

# 初始化激光束路径和光斑
laser_line, = ax_3d.plot([], [], [], 'r-', label='Laser Beam', linewidth=2)
laser_point_3d, = ax_3d.plot([], [], [], 'ro', label='Laser Spot', markersize=8)
ax_3d.legend()

# 2D 子图：显示平面上的光斑、痕迹和角度
ax_2d = fig.add_subplot(122)
ax_2d.set_xlim(-X/2 - 0.5, X/2 + 0.5)
ax_2d.set_ylim(-Y/2 - 0.5, Y/2 + 0.5)
ax_2d.set_xlabel('X')
ax_2d.set_ylabel('Y')
ax_2d.set_title('2D Laser Hanzi Drawing')
ax_2d.grid(True)
ax_2d.set_aspect('equal')

# 绘制矩形边界
rect_2d = plt.Rectangle((-X/2, -Y/2), X, Y, fill=False, edgecolor='blue', linewidth=2)
ax_2d.add_patch(rect_2d)

# 初始化激光光斑（圆形）
spot_circle = plt.Circle((0, 0), spot_size/2, color='red', alpha=0.5, label='Laser Spot')
ax_2d.add_patch(spot_circle)

# 初始化扫描痕迹（红色）
trace_line, = ax_2d.plot([], [], 'r-', alpha=0.3, label='Hanzi Path')
ax_2d.legend()

# 添加步进电机角度显示文本
alpha_text = ax_2d.text(0.05, 0.95, 'Y Motor (α): 0.00°', transform=ax_2d.transAxes, fontsize=10)
beta_text = ax_2d.text(0.05, 0.90, 'X Motor (β): 0.00°', transform=ax_2d.transAxes, fontsize=10)

# 获取自定义路径
custom_points = get_custom_path()
frame_count = len(custom_points)

# 存储扫描痕迹的坐标
trace_x = []
trace_y = []

# 动画状态
paused = False
current_frame = 0

# 动画初始化函数
def init():
    laser_line.set_data_3d([], [], [])
    laser_point_3d.set_data_3d([], [], [])
    spot_circle.center = (0, 0)
    trace_line.set_data([], [])
    alpha_text.set_text('Y Motor (α): 0.00°')
    beta_text.set_text('X Motor (β): 0.00°')
    trace_x.clear()
    trace_y.clear()
    return laser_line, laser_point_3d, spot_circle, trace_line, alpha_text, beta_text

# 动画更新函数
def update(frame):
    global current_frame
    if not paused:
        current_frame = frame % frame_count
        x, y = custom_points[current_frame]
        # 限制坐标在矩形区域内
        x = np.clip(x, -X/2, X/2)
        y = np.clip(y, -Y/2, Y/2)
        alpha, beta = xy_to_angles(x, y)
        
        # 更新 3D 激光束路径和点
        laser_x = [0, x]
        laser_y = [0, y]
        laser_z = [0, d]
        laser_line.set_data_3d(laser_x, laser_y, laser_z)
        laser_point_3d.set_data_3d([x], [y], [d])
        
        # 更新 2D 光斑（圆形）
        spot_circle.center = (x, y)
        
        # 更新扫描痕迹
        trace_x.append(x)
        trace_y.append(y)
        trace_line.set_data(trace_x, trace_y)
        
        # 更新步进电机角度显示（以度为单位）
        alpha_deg = np.degrees(alpha)
        beta_deg = np.degrees(beta)
        alpha_text.set_text(f'Y Motor (α): {alpha_deg:.2f}°')
        beta_text.set_text(f'X Motor (β): {beta_deg:.2f}°')
    
    return laser_line, laser_point_3d, spot_circle, trace_line, alpha_text, beta_text

# 键盘事件处理：按空格键暂停/继续
def on_key_press(event):
    global paused
    if event.key == ' ':
        paused = not paused
        if paused:
            print("Animation paused.")
        else:
            print("Animation resumed.")
            # 强制重绘当前帧
            x, y = custom_points[current_frame]
            x = np.clip(x, -X/2, X/2)
            y = np.clip(y, -Y/2, Y/2)
            alpha, beta = xy_to_angles(x, y)
            laser_line.set_data_3d([0, x], [0, y], [0, d])
            laser_point_3d.set_data_3d([x], [y], [d])
            spot_circle.center = (x, y)
            trace_line.set_data(trace_x, trace_y)
            alpha_deg = np.degrees(alpha)
            beta_deg = np.degrees(beta)
            alpha_text.set_text(f'Y Motor (α): {alpha_deg:.2f}°')
            beta_text.set_text(f'X Motor (β): {beta_deg:.2f}°')
            fig.canvas.draw_idle()

# 绑定键盘事件
fig.canvas.mpl_connect('key_press_event', on_key_press)

# 创建动画，禁用 blit 以确保渲染完整
ani = FuncAnimation(fig, update, frames=np.arange(0, frame_count), init_func=init, blit=False, interval=50)

# 显示动画
plt.tight_layout()
plt.show()