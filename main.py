import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 参数设置
d = 10.0  # 平面距离 z = d
X = 4.0   # 矩形宽
Y = 3.0   # 矩形高
spot_size = 0.2  # 激光光斑的直径（圆形）
cycles = 3  # 正弦波周期数

# 计算步进电机角度的最大值
alpha_max = np.arctan(Y / (2 * d))  # YZ 平面角度（y 方向电机）
beta_max = np.arctan(X / (2 * d))   # XZ 平面角度（x 方向电机）

# 定义正弦波扫描路径
def get_sine_points(steps=300):
    x = np.linspace(-X/2, X/2, steps)
    # y = (Y/2) * sin(2π * cycles * x/X)
    y = (Y/2) * np.sin(2 * np.pi * cycles * x / X)
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
ax_2d.set_title('2D Laser Spot and Path')
ax_2d.grid(True)
ax_2d.set_aspect('equal')

# 绘制矩形边界
rect_2d = plt.Rectangle((-X/2, -Y/2), X, Y, fill=False, edgecolor='blue', linewidth=2)
ax_2d.add_patch(rect_2d)

# 初始化激光光斑（圆形）
spot_circle = plt.Circle((0, 0), spot_size/2, color='red', alpha=0.5, label='Laser Spot')
ax_2d.add_patch(spot_circle)

# 初始化扫描痕迹（红色）
trace_line, = ax_2d.plot([], [], 'r-', alpha=0.3, label='Scan Path')  # 红色淡化线条
ax_2d.legend()

# 添加步进电机角度显示文本
alpha_text = ax_2d.text(0.05, 0.95, 'Y Motor (α): 0.00°', transform=ax_2d.transAxes, fontsize=10)
beta_text = ax_2d.text(0.05, 0.90, 'X Motor (β): 0.00°', transform=ax_2d.transAxes, fontsize=10)

# 获取正弦波扫描点
sine_points = get_sine_points()
frame_count = len(sine_points)

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
        x, y = sine_points[current_frame]
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
            x, y = sine_points[current_frame]
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