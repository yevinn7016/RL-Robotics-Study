# week01_fk.py
# 목표: UR5 6축 로봇 Forward Kinematics 구현
# 입력: 관절 각도 6개 (라디안)
# 출력:
# 1. 특정 관절각의 엔드이펙터 4x4 변환행렬
# 2. 전체 관절 움직임에 따른 3D 경로 gif
# 3. 특정 관절만 움직였을 때 궤적 비교 이미지

import numpy as np
import roboticstoolbox as rtb
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# 1. UR5 모델 로드(rtb.models.UR5())
try:
    robot = rtb.models.UR5()
except AttributeError:
    robot = rtb.models.DH.UR5()

print("UR5 model loaded successfully")


#FK 테스트
q_test = np.array([0, -np.pi/4, np.pi/3, 0, np.pi/6, 0])

T = robot.fkine(q_test) #fkine()은 Forward Kinematics를 계산하는 함수

print("\n=== End-Effector 4x4 Transformation Matrix ===")
print(T.A)   # SE(3) 4x4 행렬 출력

print("\n=== End-Effector XYZ Position ===")
print(T.t)   # x, y, z 위치 출력


# 2. 임의 관절 각도 50개 시퀀스 생성(np.linspace)
n_steps = 50 #움직임을 50단게로 나누자

q_start = np.array([0, 0, 0, 0, 0, 0]) #시작자세
q_end = np.array([ #끝자세
    np.pi / 2,
    -np.pi / 4,
    np.pi / 3,
    -np.pi / 6,
    np.pi / 4,
    np.pi / 2
])

q_sequence = np.linspace(q_start, q_end, n_steps) # 시작자세에서 끝자세까지 중간 관절각 50개 만듦



# 3. 각 포즈의 엔드이펙터 xyz 계산
positions = []

for q in q_sequence: #50개의 관절각마다 FK를 계산 
    T = robot.fkine(q)
    xyz = T.t
    positions.append(xyz)

positions = np.array(positions)
 
# 저장된 위치에서 x, y, z를 따로 분리 
x = positions[:, 0]
y = positions[:, 1]
z = positions[:, 2]



# 4. 3D 경로 애니메이션 gif 저장
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.set_title("UR5 End-Effector Path")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_xlim(np.min(x) - 0.1, np.max(x) + 0.1)
ax.set_ylim(np.min(y) - 0.1, np.max(y) + 0.1)
ax.set_zlim(np.min(z) - 0.1, np.max(z) + 0.1)

line, = ax.plot([], [], [], linewidth=2)
point, = ax.plot([], [], [], marker="o")


def update(frame):
    line.set_data(x[:frame + 1], y[:frame + 1])
    line.set_3d_properties(z[:frame + 1])

    point.set_data([x[frame]], [y[frame]])
    point.set_3d_properties([z[frame]])

    return line, point


ani = FuncAnimation(
    fig,
    update,
    frames=n_steps,
    interval=100,
    blit=False
)

ani.save("ur5_fk_path.gif", writer=PillowWriter(fps=10))
plt.close()

print("\nSaved: ur5_fk_path.gif")



# 5. 특정 관절만 움직일 때 궤적 비교
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.set_title("End-Effector Trajectory by Single Joint Motion")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

for joint_idx in range(6):
    q_base = np.zeros(6)
    joint_positions = []

    angles = np.linspace(-np.pi / 2, np.pi / 2, n_steps)

    for angle in angles:
        q = q_base.copy()
        q[joint_idx] = angle

        T = robot.fkine(q)
        joint_positions.append(T.t)

    joint_positions = np.array(joint_positions)

    ax.plot(
        joint_positions[:, 0],
        joint_positions[:, 1],
        joint_positions[:, 2],
        label=f"Joint {joint_idx + 1}"
    )

ax.legend()
plt.savefig("ur5_joint_trajectory_compare.png", dpi=300)
plt.show()

print("Saved: ur5_joint_trajectory_compare.png")