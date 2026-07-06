import numpy as np
from scipy.spatial.transform import Rotation as R

np.set_printoptions(precision=4, suppress = True)

##1. 오일러각 -> 쿼터니언 & 회전행렬 변환
euler_deg = [30, 45, 60]

r = R.from_euler("xyz", euler_deg, degrees=True)

# scipy는 [x, y, z, w] 순서
quat = r.as_quat()     # 쿼터니언 변환 함수
matrix = r.as_matrix() # 회전행렬 변환 함수

print("입력 Euler angle [roll, pitch, yaw] =", euler_deg)
print("Quaternion [x, y, z, w] =")
print(quat)

print("Rotation Matrix =")
print(matrix)

# 2. 회전행렬 -> 쿼터니언 & 오일러각 변환
r_from_matrix = R.from_matrix(matrix) # matrix.as_quat()는 불가능 -> from_matrix함수 사용해서 Rotation 객체로 변환 

print("Matrix에서 다시 Quaternion =")
print(r_from_matrix.as_quat())

print("Matrix에서 다시 Euler angle =")
print(r_from_matrix.as_euler("xyz", degrees=True))

# 3. 쿼터니언 -> 오일러 & 회전행렬 변환
r_from_quat = R.from_quat(quat)

print("Quaternion에서 다시 Euler angle =")
print(r_from_quat.as_euler("xyz", degrees=True))

print("Quaternion에서 다시 Rotation Matrix =")
print(r_from_quat.as_matrix())