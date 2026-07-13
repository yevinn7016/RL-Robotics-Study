"""
week02_ik.py

Part A
- UR5 수치 IK 구현
- Newton-Raphson 반복
- Damped Least Squares(DLS)
- 특이점 감지
- 최대 반복 100회
- tolerance = 1e-4

Part B
- RoboSuite Lift 환경 실행
- 랜덤 액션으로 에피소드 1회 실행
- 매 스텝의 관절각도, 엔드이펙터 위치, 보상을 CSV로 저장
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


# 공통 설정
np.set_printoptions(precision=6, suppress=True)

OUTPUT_DIR = Path(__file__).resolve().parent
CSV_PATH = OUTPUT_DIR / "robosuite_lift_log.csv"


# Part A: UR5 수치 IK
def damped_pseudoinverse(
    jacobian: np.ndarray,
    damping: float,
) -> np.ndarray:
    """
    DLS 의사역행렬 계산

    J⁺ = Jᵀ (J Jᵀ + λ² I)⁻¹

    실제 구현에서는 수치 안정성을 위해 inverse 대신 solve를 사용
    """
    task_dimension = jacobian.shape[0]

    regularized_matrix = (
        jacobian @ jacobian.T
        + (damping ** 2) * np.eye(task_dimension)
    )

    # J.T @ inverse(regularized_matrix)와 같은 결과
    return np.linalg.solve(
        regularized_matrix,
        jacobian,
    ).T


def numerical_ik_dls(
    robot: Any,
    target_position: np.ndarray,
    initial_q: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    damping: float = 0.05,
    singularity_threshold: float = 1e-8,
    max_joint_step: float = 0.2,
) -> tuple[np.ndarray, bool, int]:
    """
    UR5 엔드이펙터의 위치만 맞추는 수치 IK.

    Parameters
    ----------
    robot:
        Robotics Toolbox의 UR5 모델
    target_position:
        목표 엔드이펙터 위치 [x, y, z]
    initial_q:
        초기 관절각 6개 [rad]
    max_iterations:
        최대 반복 횟수
    tolerance:
        위치 오차 허용값 [m]
    damping:
        DLS 감쇠 계수 λ
    singularity_threshold:
        det(JJᵀ)가 이 값보다 작으면 특이점 근처로 판단
    max_joint_step:
        한 번의 반복에서 허용할 최대 관절 변화량 [rad]

    Returns
    -------
    q:
        계산된 관절각
    success:
        수렴 여부
    iteration:
        종료된 반복 횟수
    """
    q = np.asarray(initial_q, dtype=float).copy()
    target_position = np.asarray(target_position, dtype=float).reshape(3)

    print("\n" + "=" * 65)
    print("Part A: UR5 Numerical IK with DLS")
    print("=" * 65)
    print(f"목표 위치: {target_position}")
    print(f"초기 관절각: {q}")
    print(f"Damping λ: {damping}")
    print(f"Tolerance: {tolerance}")

    for iteration in range(max_iterations):
        # ----------------------------------------------------
        # 1. 현재 관절각으로 FK 계산
        # ----------------------------------------------------
        current_transform = robot.fkine(q)
        current_position = np.asarray(current_transform.t).reshape(3)

        # ----------------------------------------------------
        # 2. 목표 위치와 현재 위치의 오차 계산
        # e = target - fk(q)
        # ----------------------------------------------------
        error = target_position - current_position
        error_norm = np.linalg.norm(error)

        print(
            f"반복 {iteration:03d} | "
            f"현재 위치 = {current_position} | "
            f"오차 = {error_norm:.8f}"
        )

        # ----------------------------------------------------
        # 3. 오차가 충분히 작으면 종료
        # ----------------------------------------------------
        if error_norm < tolerance:
            print("\nIK 수렴 성공!")
            print(f"반복 횟수: {iteration}")
            print(f"최종 위치 오차: {error_norm:.8f} m")

            return q, True, iteration

        # ----------------------------------------------------
        # 4. Jacobian 계산
        #
        # robot.jacob0(q)는 일반적으로 6×6 Jacobian
        # 앞의 3개 행은 x, y, z 선속도 부분
        # ----------------------------------------------------
        full_jacobian = robot.jacob0(q)
        position_jacobian = full_jacobian[:3, :]

        # ----------------------------------------------------
        # 5. 특이점 검사
        #
        # 위치 Jacobian은 3×6이므로
        # JJᵀ는 3×3 정방행렬
        # ----------------------------------------------------
        jjt = position_jacobian @ position_jacobian.T
        determinant = float(np.linalg.det(jjt))

        if determinant < singularity_threshold:
            print(
                "  [경고] 특이점 또는 특이점 근처입니다. "
                f"det(JJᵀ) = {determinant:.3e}"
            )

        # ----------------------------------------------------
        # 6. DLS 의사역행렬 계산
        #
        # J⁺ = Jᵀ(JJᵀ + λ²I)⁻¹
        # ----------------------------------------------------
        j_dls = damped_pseudoinverse(
            jacobian=position_jacobian,
            damping=damping,
        )

        # ----------------------------------------------------
        # 7. Newton-Raphson 관절각 업데이트
        #
        # 과제 식:
        # q_new = q - J⁺(fk(q) - target)
        #
        # error = target - fk(q)이므로 아래 식과 동일:
        # q_new = q + J⁺ error
        # ----------------------------------------------------
        delta_q = j_dls @ error

        # 지나치게 큰 관절 변화 방지
        largest_step = np.max(np.abs(delta_q))

        if largest_step > max_joint_step:
            delta_q *= max_joint_step / largest_step

        q = q + delta_q

        # 각도를 -π ~ π 범위로 정규화
        q = (q + np.pi) % (2.0 * np.pi) - np.pi

    # 최대 반복 횟수까지 수렴하지 못한 경우
    final_position = np.asarray(robot.fkine(q).t).reshape(3)
    final_error = np.linalg.norm(target_position - final_position)

    print("\nIK 수렴 실패")
    print(f"최대 반복 횟수 {max_iterations}회에 도달했습니다.")
    print(f"최종 위치: {final_position}")
    print(f"최종 위치 오차: {final_error:.8f} m")

    return q, False, max_iterations


def run_part_a() -> None:
    """UR5 수치 IK 예제를 실행한다."""
    try:
        import roboticstoolbox as rtb
    except ImportError as exc:
        raise RuntimeError(
            "roboticstoolbox가 설치되어 있지 않습니다.\n"
            "다음 명령어로 설치하세요:\n"
            "pip install roboticstoolbox-python spatialmath-python"
        ) from exc

    # Robotics Toolbox의 DH 기반 UR5 모델
    robot = rtb.models.DH.UR5()

    # --------------------------------------------------------
    # 도달 가능한 목표 위치 생성
    #
    # 임의의 목표 관절각으로 FK를 먼저 계산하여
    # 목표 위치가 반드시 UR5 작업공간 안에 있도록 한다.
    # --------------------------------------------------------
    target_q_for_test = np.array(
        [0.3, -1.0, 1.1, -0.5, 0.8, 0.2],
        dtype=float,
    )

    target_position = np.asarray(
        robot.fkine(target_q_for_test).t
    ).reshape(3)

    # 수치 IK가 시작할 초기 추측값
    initial_q = np.array(
        [0.0, -0.7, 0.7, 0.0, 0.5, 0.0],
        dtype=float,
    )

    solution_q, success, iterations = numerical_ik_dls(
        robot=robot,
        target_position=target_position,
        initial_q=initial_q,
        max_iterations=100,
        tolerance=1e-4,
        damping=0.05,
        singularity_threshold=1e-8,
        max_joint_step=0.2,
    )

    final_position = np.asarray(
        robot.fkine(solution_q).t
    ).reshape(3)

    print("\n--- Part A 최종 결과 ---")
    print(f"수렴 여부: {success}")
    print(f"반복 횟수: {iterations}")
    print(f"계산된 관절각 [rad]: {solution_q}")
    print(f"계산된 관절각 [deg]: {np.rad2deg(solution_q)}")
    print(f"목표 위치: {target_position}")
    print(f"최종 위치: {final_position}")
    print(
        "최종 오차:",
        np.linalg.norm(target_position - final_position),
    )


# ============================================================
# Part B: RoboSuite Lift 환경
# ============================================================

def find_observation(
    observation: dict[str, np.ndarray],
    possible_keys: list[str],
) -> np.ndarray | None:
    """
    RoboSuite 버전에 따라 관측 key 이름이 다를 수 있으므로
    여러 후보 중 존재하는 값을 찾는다.
    """
    for key in possible_keys:
        if key in observation:
            return np.asarray(observation[key], dtype=float).flatten()

    return None


def get_joint_positions(
    env: Any,
    observation: dict[str, np.ndarray],
) -> np.ndarray:
    """
    우선 observation에서 관절각을 찾고,
    없으면 MuJoCo qpos에서 직접 읽는다.
    """
    joint_positions = find_observation(
        observation,
        [
            "robot0_joint_pos",
            "robot0_arm_joint_pos",
            "robot0_joint_positions",
        ],
    )

    if joint_positions is not None:
        return joint_positions

    # 관측 딕셔너리에 개별 관절각이 없는 경우
    try:
        robot = env.robots[0]
        joint_indexes = robot._ref_joint_pos_indexes

        return np.asarray(
            env.sim.data.qpos[joint_indexes],
            dtype=float,
        ).flatten()

    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "RoboSuite에서 로봇 관절각을 읽지 못했습니다.\n"
            f"현재 observation key: {list(observation.keys())}"
        ) from exc


def get_end_effector_position(
    env: Any,
    observation: dict[str, np.ndarray],
) -> np.ndarray:
    """
    우선 observation에서 엔드이펙터 위치를 찾고,
    없으면 MuJoCo body/site 위치에서 직접 읽는다.
    """
    eef_position = find_observation(
        observation,
        [
            "robot0_eef_pos",
            "robot0_right_eef_pos",
            "robot0_gripper_pos",
        ],
    )

    if eef_position is not None:
        return eef_position[:3]

    # 관측에 없다면 로봇 객체의 eef site를 통해 읽는다.
    try:
        robot = env.robots[0]

        # RoboSuite 버전에 따라 eef_site_id 형식이 다를 수 있다.
        eef_site_id = robot.eef_site_id

        if isinstance(eef_site_id, dict):
            eef_site_id = next(iter(eef_site_id.values()))

        return np.asarray(
            env.sim.data.site_xpos[eef_site_id],
            dtype=float,
        ).reshape(3)

    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "RoboSuite에서 엔드이펙터 위치를 읽지 못했습니다.\n"
            f"현재 observation key: {list(observation.keys())}"
        ) from exc


def build_csv_header(
    joint_count: int,
) -> list[str]:
    """관절 개수에 맞춰 CSV 헤더를 만든다."""
    joint_columns = [
        f"joint_{index + 1}_rad"
        for index in range(joint_count)
    ]

    return [
        "step",
        "simulation_time",
        *joint_columns,
        "eef_x",
        "eef_y",
        "eef_z",
        "reward",
        "done",
    ]


def run_part_b(
    csv_path: Path = CSV_PATH,
    horizon: int = 200,
    render: bool = True,
) -> None:
    """RoboSuite Lift 환경에서 랜덤 액션 에피소드 1회를 실행한다."""
    try:
        import robosuite as suite
    except ImportError as exc:
        raise RuntimeError(
            "robosuite가 설치되어 있지 않습니다.\n"
            "다음 명령어로 설치하세요:\n"
            "pip install mujoco robosuite"
        ) from exc

    print("\n" + "=" * 65)
    print("Part B: RoboSuite Lift Random Episode")
    print("=" * 65)

    env = None

    try:
        # ----------------------------------------------------
        # 1. Lift 환경 생성
        # ----------------------------------------------------
        env = suite.make(
            env_name="Lift",
            robots="Panda",
            has_renderer=render,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            use_object_obs=True,
            reward_shaping=True,
            control_freq=20,
            horizon=horizon,
            ignore_done=False,
        )

        # ----------------------------------------------------
        # 2. 환경 초기화
        # ----------------------------------------------------
        observation = env.reset()

        print("환경 생성 완료")
        print(f"Observation keys: {list(observation.keys())}")

        # 랜덤 액션의 최솟값과 최댓값
        action_low, action_high = env.action_spec

        print(f"Action shape: {action_low.shape}")
        print(f"Action low: {action_low}")
        print(f"Action high: {action_high}")

        # 첫 상태를 이용해 관절 개수 확인
        initial_joint_positions = get_joint_positions(
            env,
            observation,
        )

        csv_header = build_csv_header(
            joint_count=len(initial_joint_positions),
        )

        total_reward = 0.0
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------
        # 3. CSV 파일 생성
        # ----------------------------------------------------
        with csv_path.open(
            mode="w",
            newline="",
            encoding="utf-8-sig",
        ) as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(csv_header)

            done = False
            step = 0

            # ------------------------------------------------
            # 4. 에피소드 한 번 실행
            # ------------------------------------------------
            while not done and step < horizon:
                # action 범위 안에서 랜덤 액션 생성
                action = np.random.uniform(
                    low=action_low,
                    high=action_high,
                )

                observation, reward, done, info = env.step(action)

                if render:
                    env.render()

                joint_positions = get_joint_positions(
                    env,
                    observation,
                )

                eef_position = get_end_effector_position(
                    env,
                    observation,
                )

                simulation_time = float(env.sim.data.time)
                total_reward += float(reward)

                row = [
                    step,
                    simulation_time,
                    *joint_positions.tolist(),
                    float(eef_position[0]),
                    float(eef_position[1]),
                    float(eef_position[2]),
                    float(reward),
                    bool(done),
                ]

                writer.writerow(row)

                if step % 20 == 0:
                    print(
                        f"step={step:03d} | "
                        f"eef={eef_position} | "
                        f"reward={reward:.6f}"
                    )

                step += 1

        print("\n--- Part B 최종 결과 ---")
        print(f"실행 스텝 수: {step}")
        print(f"누적 보상: {total_reward:.6f}")
        print(f"CSV 저장 위치: {csv_path.resolve()}")

    finally:
        # 오류가 발생해도 환경을 닫는다.
        if env is not None:
            env.close()


# ============================================================
# 메인 실행
# ============================================================

def main() -> None:
    try:
        run_part_a()
    except Exception as exc:
        print("\n[Part A 실행 오류]")
        print(exc)

    try:
        run_part_b(
            csv_path=CSV_PATH,
            horizon=200,
            render=True,
        )
    except Exception as exc:
        print("\n[Part B 실행 오류]")
        print(exc)


if __name__ == "__main__":
    main()