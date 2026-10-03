# src/engines/math_engine.py

import numpy as np
import polars as pl

class MathEngine:
    """
    PX4 로그 분석을 위한 가상 시그널(Virtual Signals) 및 주파수 변환 엔진
    """
    
    @staticmethod
    def preprocess_dataset(dataset):
        """로그 파일이 로드된 직후, 제어 분석에 필수적인 가상 데이터를 일괄 생성합니다."""
        
        # 1. Attitude (Quaternion -> Euler [deg])
        # 1-1. Actual (실제 자세)
        if "vehicle_attitude_0" in dataset.topics:
            topic = dataset.topics["vehicle_attitude_0"]
            df = topic.dataframe
            if "q[0]" in df.columns:
                q0, q1, q2, q3 = df["q[0]"].to_numpy(), df["q[1]"].to_numpy(), df["q[2]"].to_numpy(), df["q[3]"].to_numpy()
                roll = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2)) * 57.2958
                pitch = np.arcsin(np.clip(2*(q0*q2 - q3*q1), -1.0, 1.0)) * 57.2958
                yaw = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2)) * 57.2958
                
                topic.dataframe = df.with_columns([
                    pl.Series("roll_euler", roll), pl.Series("pitch_euler", pitch), pl.Series("yaw_euler", yaw)
                ])
                for s in ["roll_euler", "pitch_euler", "yaw_euler"]: topic.signals[s] = None

        # 1-2. Setpoint (목표 자세)
        if "vehicle_attitude_setpoint_0" in dataset.topics:
            topic = dataset.topics["vehicle_attitude_setpoint_0"]
            df = topic.dataframe
            if "q_d[0]" in df.columns:
                q0, q1, q2, q3 = df["q_d[0]"].to_numpy(), df["q_d[1]"].to_numpy(), df["q_d[2]"].to_numpy(), df["q_d[3]"].to_numpy()
                roll = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2)) * 57.2958
                pitch = np.arcsin(np.clip(2*(q0*q2 - q3*q1), -1.0, 1.0)) * 57.2958
                yaw = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2)) * 57.2958

                topic.dataframe = df.with_columns([
                    pl.Series("roll_sp_euler", roll), pl.Series("pitch_sp_euler", pitch), pl.Series("yaw_sp_euler", yaw)
                ])
                for s in ["roll_sp_euler", "pitch_sp_euler", "yaw_sp_euler"]: topic.signals[s] = None
            elif "pitch_body" in df.columns:
                # (2026-05-28) q_d 없이 *_body 만 있는 구형 PX4 ulg fallback.
                # actual(*_euler) 은 deg 로 저장되므로 단위 일치를 위해 rad → deg 변환해서 *_sp_euler 로 통일.
                roll  = df["roll_body"].to_numpy()  * 57.2958
                pitch = df["pitch_body"].to_numpy() * 57.2958
                yaw   = df["yaw_body"].to_numpy()   * 57.2958

                topic.dataframe = df.with_columns([
                    pl.Series("roll_sp_euler", roll), pl.Series("pitch_sp_euler", pitch), pl.Series("yaw_sp_euler", yaw)
                ])
                for s in ["roll_sp_euler", "pitch_sp_euler", "yaw_sp_euler"]: topic.signals[s] = None

        # 2. Altitude (NED Z -> Up [m])
        if "vehicle_local_position_0" in dataset.topics:
            topic = dataset.topics["vehicle_local_position_0"]
            df = topic.dataframe
            if "z" in df.columns:
                topic.dataframe = df.with_columns(pl.Series("alt_up", -df["z"].to_numpy()))
                topic.signals["alt_up"] = None

        if "vehicle_local_position_setpoint_0" in dataset.topics:
            topic = dataset.topics["vehicle_local_position_setpoint_0"]
            df = topic.dataframe
            if "z" in df.columns:
                topic.dataframe = df.with_columns(pl.Series("alt_sp_up", -df["z"].to_numpy()))
                topic.signals["alt_sp_up"] = None

        # 3. Ground Speed (sqrt(vx^2 + vy^2) [m/s])
        if "vehicle_local_position_0" in dataset.topics:
            topic = dataset.topics["vehicle_local_position_0"]
            df = topic.dataframe
            if "vx" in df.columns and "vy" in df.columns:
                vx, vy = df["vx"].to_numpy(), df["vy"].to_numpy()
                gs = np.sqrt(vx**2 + vy**2)
                topic.dataframe = df.with_columns(pl.Series("ground_speed_mag", gs))
                topic.signals["ground_speed_mag"] = None

        # 4. Cumulative Flight Distance (2D / 3D) — Analysis → Flight Path → 2D/3D Distance.
        # (2026-05-28) 인접 위치 샘플 간 세그먼트 길이를 누적. NaN 세그먼트는 0으로 무시해
        # 데이터 결손 후에도 cum 곡선이 NaN 전파로 끊기지 않게 함.
        if "vehicle_local_position_0" in dataset.topics:
            topic = dataset.topics["vehicle_local_position_0"]
            df = topic.dataframe
            if "x" in df.columns and "y" in df.columns:
                x = df["x"].to_numpy().astype("float64")
                y = df["y"].to_numpy().astype("float64")
                dx = np.diff(x, prepend=x[0] if len(x) else 0.0)
                dy = np.diff(y, prepend=y[0] if len(y) else 0.0)
                seg2d = np.sqrt(dx * dx + dy * dy)
                seg2d = np.nan_to_num(seg2d, nan=0.0, posinf=0.0, neginf=0.0)
                cum_2d = np.cumsum(seg2d)
                new_cols = [pl.Series("cum_distance_2d", cum_2d)]
                if "z" in df.columns:
                    z = df["z"].to_numpy().astype("float64")
                    dz = np.diff(z, prepend=z[0] if len(z) else 0.0)
                    seg3d = np.sqrt(dx * dx + dy * dy + dz * dz)
                    seg3d = np.nan_to_num(seg3d, nan=0.0, posinf=0.0, neginf=0.0)
                    cum_3d = np.cumsum(seg3d)
                    new_cols.append(pl.Series("cum_distance_3d", cum_3d))
                topic.dataframe = df.with_columns(new_cols)
                for s in ("cum_distance_2d", "cum_distance_3d"):
                    if s in topic.dataframe.columns:
                        topic.signals[s] = None

        # 5. Actuator PWM (control[i] ∈ [0,1] → 1000~2000us 매핑) — Analysis → Actuator → VTOL/MC / F/W.
        # (2026-05-28) PX4 actuator_motors.control[i] 는 0~1 정규화 값. 사용자가 PWM(us) 단위로 보길
        # 원해 1000+1000*c 로 변환된 control_pwm[i] 컬럼을 미리 만들어 둠. (2026-05-31) F/W 분석용으로
        # actuator_servos_0 도 같은 변환 적용 — F/W 핸들러가 모터+서보 2행 비교에 사용.
        for topic_name in ("actuator_motors_0", "actuator_servos_0"):
            if topic_name not in dataset.topics:
                continue
            topic = dataset.topics[topic_name]
            df = topic.dataframe
            pwm_cols = []
            for i in range(12):  # PX4 actuator_* 슬롯 최대 12 (motors/servos 둘 다 충분).
                col = f"control[{i}]"
                if col not in df.columns:
                    continue
                vals = df[col].to_numpy().astype("float64")
                # NaN 은 NaN 유지 (disarmed/no-output 구간 표시 안 됨).
                pwm = 1000.0 + 1000.0 * vals
                pwm_cols.append(pl.Series(f"control_pwm[{i}]", pwm))
            if pwm_cols:
                topic.dataframe = df.with_columns(pwm_cols)
                for s in pwm_cols:
                    topic.signals[s.name] = None

    @staticmethod
    def compute_fft(time_sec, data):
        """
        시간 도메인 데이터를 Hanning Window 적용 후 주파수 도메인(FFT)으로 변환합니다.
        반환값: (frequency_array, amplitude_array)
        """
        valid_idx = ~np.isnan(data)
        t = time_sec[valid_idx]
        y = data[valid_idx]
        
        if len(t) < 2: return np.array([]), np.array([])
        
        # 평균 샘플링 주기 계산
        dt = np.mean(np.diff(t))
        if dt <= 0: return np.array([]), np.array([])
        
        N = len(y)
        # 누설 오차 방지를 위한 Hanning Window
        window = np.hanning(N)
        y_windowed = y * window
        
        yf = np.fft.fft(y_windowed)
        xf = np.fft.fftfreq(N, d=dt)
        
        # 양의 주파수 영역만 추출하여 진폭 보정
        idx = np.where(xf >= 0)
        amp = 2.0 / N * np.abs(yf[idx])
        
        return xf[idx], amp