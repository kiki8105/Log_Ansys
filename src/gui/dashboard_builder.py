# src/gui/dashboard_builder.py

import os
import sys
import numpy as np
import polars as pl

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from analysis.detector import FlightTypeDetector

class DashboardBuilder:
    def __init__(self, workspace, dataset, file_name):
        self.workspace = workspace
        self.dataset = dataset
        self.file_name = file_name

    def _find_topic_by_prefix(self, prefix):
        if prefix in self.dataset.topics:
            return prefix
        for topic_name in self.dataset.topics.keys():
            if topic_name.startswith(prefix):
                return topic_name
        return None

    @staticmethod
    def _find_first_col(df_columns, candidates):
        for col in candidates:
            if col in df_columns:
                return col
        return None

    @staticmethod
    def _fit_xy_range(plot_widget, x_arr, y_arr, pad_ratio=0.05):
        if len(x_arr) == 0 or len(y_arr) == 0:
            return
        try:
            x_num = np.asarray(x_arr, dtype=np.float64)
            y_num = np.asarray(y_arr, dtype=np.float64)
        except Exception:
            return
        mask = np.isfinite(x_num) & np.isfinite(y_num)
        if not np.any(mask):
            return
        x_valid = x_num[mask]
        y_valid = y_num[mask]
        x_min, x_max = float(np.min(x_valid)), float(np.max(x_valid))
        y_min, y_max = float(np.min(y_valid)), float(np.max(y_valid))
        if x_min == x_max:
            x_min -= 1.0
            x_max += 1.0
        if y_min == y_max:
            y_min -= 1.0
            y_max += 1.0
        x_pad = (x_max - x_min) * pad_ratio
        y_pad = (y_max - y_min) * pad_ratio
        plot_widget.plot.setXRange(x_min - x_pad, x_max + x_pad, padding=0)
        plot_widget.plot.setYRange(y_min - y_pad, y_max + y_pad, padding=0)

    @staticmethod
    def _add_binary_flag(df, src_col, dst_col):
        if src_col not in df.columns:
            return df, False
        df = df.with_columns(
            pl.when(pl.col(src_col).is_null())
            .then(None)
            .otherwise((pl.col(src_col).cast(pl.Float64).abs() > 0.5).cast(pl.Float64))
            .alias(dst_col)
        )
        return df, True

    def build(self):
        print(f"\n[Dashboard] '{self.file_name}' 분석을 위한 6행 기본 대시보드 구성을 시작합니다.")
        
        detector = FlightTypeDetector(self.dataset)
        airframe = detector.detect()

        self.workspace.create_grid([2, 2, 2, 2, 2, 1])
        
        p_path = self.workspace.get_plot(0, 0)
        p_alt  = self.workspace.get_plot(0, 1)
        
        p_roll_ang  = self.workspace.get_plot(1, 0)
        p_roll_rate = self.workspace.get_plot(1, 1)
        
        p_pitch_ang  = self.workspace.get_plot(2, 0)
        p_pitch_rate = self.workspace.get_plot(2, 1)
        
        p_yaw_ang  = self.workspace.get_plot(3, 0)
        p_yaw_rate = self.workspace.get_plot(3, 1)
        
        p_speed = self.workspace.get_plot(4, 0)
        p_fft   = self.workspace.get_plot(4, 1)
        
        p_spec  = self.workspace.get_plot(5, 0)

        # 사용자 요청 파스텔톤 컬러 매핑
        COLOR_SP = "#FF6B6B"   
        COLOR_ACT = "#4DA3FF"  

        # ---------------------------------------------------------
        # [1행] 2D Flight Path (정사각 매핑) | Altitude (Up=Positive)
        # ---------------------------------------------------------
        p_path.plot.setTitle("2D Flight Path (North-East)")
        # 대시보드 레이아웃이 가로로 긴 형태라 Equal Axis를 고정하면 경로가 눌려 보일 수 있어 해제
        p_path.plot.setAspectLocked(False)
        p_path.plot.setLabel('bottom', 'East')
        p_path.plot.setLabel('left', 'North')

        path_drawn = False
        local_pos_topic = self._find_topic_by_prefix("vehicle_local_position")
        if local_pos_topic:
            cols = self.dataset.topics[local_pos_topic].dataframe.columns
            north_col = self._find_first_col(cols, ["x", "north", "position[0]"])
            east_col = self._find_first_col(cols, ["y", "east", "position[1]"])
            if north_col and east_col:
                # X축=East, Y축=North
                p_path.render_signal(self.file_name, local_pos_topic, north_col, x_axis_col=east_col, color=COLOR_ACT)
                x_arr = self.dataset.topics[local_pos_topic].dataframe[east_col].to_numpy()
                y_arr = self.dataset.topics[local_pos_topic].dataframe[north_col].to_numpy()
                self._fit_xy_range(p_path, x_arr, y_arr)
                path_drawn = True

        if not path_drawn:
            global_pos_topic = self._find_topic_by_prefix("vehicle_global_position")
            if global_pos_topic:
                cols = self.dataset.topics[global_pos_topic].dataframe.columns
                lat_col = self._find_first_col(cols, ["lat", "latitude_deg", "lat_deg"])
                lon_col = self._find_first_col(cols, ["lon", "longitude_deg", "lon_deg"])
                if lat_col and lon_col:
                    p_path.plot.setTitle("2D Flight Path (Global Lat-Lon)")
                    p_path.plot.setLabel('bottom', 'Longitude')
                    p_path.plot.setLabel('left', 'Latitude')
                    p_path.render_signal(self.file_name, global_pos_topic, lat_col, x_axis_col=lon_col, color=COLOR_ACT)
                    x_arr = self.dataset.topics[global_pos_topic].dataframe[lon_col].to_numpy()
                    y_arr = self.dataset.topics[global_pos_topic].dataframe[lat_col].to_numpy()
                    self._fit_xy_range(p_path, x_arr, y_arr)
                    path_drawn = True

        if path_drawn:
            p_path.plot.enableAutoRange(x=False, y=False)
        else:
            p_path.plot.setTitle("2D Flight Path (Position Data Not Found)")

        p_alt.plot.setTitle("Altitude vs Altitude Setpoint (Up=Positive)")
        p_alt.render_signal(self.file_name, "vehicle_local_position_setpoint_0", "alt_sp_up", color=COLOR_SP)
        p_alt.render_signal(self.file_name, "vehicle_local_position_0", "alt_up", color=COLOR_ACT)

        # ---------------------------------------------------------
        # [2행] Roll Angle (Euler) | Roll Rate
        # ---------------------------------------------------------
        p_roll_ang.plot.setTitle("Roll Angle vs Roll Setpoint (deg)")
        p_roll_ang.render_signal(self.file_name, "vehicle_attitude_setpoint_0", "roll_sp_euler", color=COLOR_SP)
        p_roll_ang.render_signal(self.file_name, "vehicle_attitude_0", "roll_euler", color=COLOR_ACT)

        p_roll_rate.plot.setTitle("Roll Angular Rate vs Setpoint")
        p_roll_rate.render_signal(self.file_name, "vehicle_rates_setpoint_0", "roll", color=COLOR_SP)
        p_roll_rate.render_signal(self.file_name, "vehicle_angular_velocity_0", "xyz[0]", color=COLOR_ACT)

        # ---------------------------------------------------------
        # [3행] Pitch Angle (Euler) | Pitch Rate
        # ---------------------------------------------------------
        p_pitch_ang.plot.setTitle("Pitch Angle vs Pitch Setpoint (deg)")
        p_pitch_ang.render_signal(self.file_name, "vehicle_attitude_setpoint_0", "pitch_sp_euler", color=COLOR_SP)
        p_pitch_ang.render_signal(self.file_name, "vehicle_attitude_0", "pitch_euler", color=COLOR_ACT)

        p_pitch_rate.plot.setTitle("Pitch Angular Rate vs Setpoint")
        p_pitch_rate.render_signal(self.file_name, "vehicle_rates_setpoint_0", "pitch", color=COLOR_SP)
        p_pitch_rate.render_signal(self.file_name, "vehicle_angular_velocity_0", "xyz[1]", color=COLOR_ACT)

        # ---------------------------------------------------------
        # [4행] Yaw Angle (Euler) | Yaw Rate
        # ---------------------------------------------------------
        p_yaw_ang.plot.setTitle("Yaw Angle vs Yaw Setpoint (deg)")
        p_yaw_ang.render_signal(self.file_name, "vehicle_attitude_setpoint_0", "yaw_sp_euler", color=COLOR_SP)
        p_yaw_ang.render_signal(self.file_name, "vehicle_attitude_0", "yaw_euler", color=COLOR_ACT)

        p_yaw_rate.plot.setTitle("Yaw Angular Rate vs Setpoint")
        p_yaw_rate.render_signal(self.file_name, "vehicle_rates_setpoint_0", "yaw", color=COLOR_SP)
        p_yaw_rate.render_signal(self.file_name, "vehicle_angular_velocity_0", "xyz[2]", color=COLOR_ACT)

        # ---------------------------------------------------------
        # [5행] Speed | FFT Vibration
        # ---------------------------------------------------------
        p_speed.plot.setTitle("True Airspeed vs GPS Ground Speed")
        p_speed.render_signal(self.file_name, "vehicle_gps_position_0", "vel_m_s", color=COLOR_ACT) 
        p_speed.render_signal(self.file_name, "tecs_status_0", "true_airspeed_filtered", color=COLOR_SP)

        # 💡 [핵심] is_fft=True 파라미터를 추가하여 렌더러가 주파수 변환을 수행하도록 지시
        p_fft.plot.setTitle("FFT Accel Vibration (X=Red, Y=Green, Z=Blue)")
        p_fft.render_signal(self.file_name, "sensor_combined_0", "accelerometer_m_s2[0]", color="#FF6B6B", is_fft=True) 
        p_fft.render_signal(self.file_name, "sensor_combined_0", "accelerometer_m_s2[1]", color="#33FF33", is_fft=True) 
        p_fft.render_signal(self.file_name, "sensor_combined_0", "accelerometer_m_s2[2]", color="#4DA3FF", is_fft=True) 

        # ---------------------------------------------------------
        # [6행] 기체 타입별 패널 (1열)
        # ---------------------------------------------------------
        if airframe == "FIXED_WING":
            p_spec.plot.setTitle("FW: TECS Throttle Trim vs Throttle Setpoint")
            p_spec.render_signal(self.file_name, "tecs_status_0", "throttle_sp", color=COLOR_SP)
            p_spec.render_signal(self.file_name, "tecs_status_0", "throttle_trim", color=COLOR_ACT)
            
        elif airframe == "VTOL":
            p_spec.plot.setTitle("VTOL: Transition / Back-Transition")
            p_spec.plot.setLabel('left', 'State (0/1)')
            p_spec.plot.setYRange(-0.1, 1.2, padding=0)

            status_topic_candidates = []
            for prefix in ("vtol_vehicle_status", "vehicle_status"):
                topic_name = self._find_topic_by_prefix(prefix)
                if topic_name and topic_name not in status_topic_candidates:
                    status_topic_candidates.append(topic_name)

            drew_transition = False
            drew_to_fw = False
            drew_back = False
            drew_mode = False
            used_estimated = False

            for status_topic in status_topic_candidates:
                status_df = self.dataset.topics[status_topic].dataframe
                status_cols = status_df.columns
                status_df, has_transition = self._add_binary_flag(
                    status_df, "in_transition_mode", "in_transition_mode_flag"
                )
                status_df, has_to_fw = self._add_binary_flag(
                    status_df, "in_transition_to_fw", "in_transition_to_fw_flag"
                )
                self.dataset.topics[status_topic].dataframe = status_df

                if has_transition and not drew_transition:
                    drew_transition = bool(
                        p_spec.render_signal(self.file_name, status_topic, "in_transition_mode_flag", color="#4DA3FF")
                    )

                if has_to_fw and not drew_to_fw:
                    drew_to_fw = bool(
                        p_spec.render_signal(self.file_name, status_topic, "in_transition_to_fw_flag", color="#FF6B6B")
                    )

                if has_transition and has_to_fw and "in_transition_back_flag" not in status_df.columns:
                    status_df = status_df.with_columns(
                        (
                            (pl.col("in_transition_mode_flag") > 0.5) &
                            (pl.col("in_transition_to_fw_flag") < 0.5)
                        ).cast(pl.Float64).alias("in_transition_back_flag")
                    )
                    self.dataset.topics[status_topic].dataframe = status_df

                if "in_transition_back_flag" in status_df.columns and not drew_back:
                    drew_back = bool(
                        p_spec.render_signal(self.file_name, status_topic, "in_transition_back_flag", color="#33CC66")
                    )

                # Fallback mode signal: 1=Fixed-Wing mode, 0=Rotary-Wing mode
                status_cols = status_df.columns
                if "is_fixed_wing_mode" not in status_cols and "vehicle_type" in status_cols:
                    status_df = status_df.with_columns(
                        (pl.col("vehicle_type").cast(pl.Float64) == 2).cast(pl.Float64).alias("is_fixed_wing_mode")
                    )
                    self.dataset.topics[status_topic].dataframe = status_df
                    status_cols = status_df.columns

                if "is_fixed_wing_mode" not in status_cols and "vtol_in_rw_mode" in status_cols:
                    status_df = status_df.with_columns(
                        (pl.col("vtol_in_rw_mode").cast(pl.Float64) < 0.5).cast(pl.Float64).alias("is_fixed_wing_mode")
                    )
                    self.dataset.topics[status_topic].dataframe = status_df
                    status_cols = status_df.columns

                if "is_fixed_wing_mode" in status_cols and not drew_mode:
                    drew_mode = bool(
                        p_spec.render_signal(self.file_name, status_topic, "is_fixed_wing_mode", color="#FFD54F")
                    )

                # If explicit transition flags are missing, estimate from mode flips.
                if not drew_transition and not drew_to_fw and not drew_back and "is_fixed_wing_mode" in status_cols:
                    mode_raw = status_df["is_fixed_wing_mode"].to_numpy()
                    mode_arr = np.asarray(mode_raw, dtype=np.float64)
                    if len(mode_arr) > 1:
                        mode_bin = np.where(np.isfinite(mode_arr) & (mode_arr > 0.5), 1.0, 0.0)
                        mode_diff = np.diff(mode_bin, prepend=mode_bin[0])
                        to_fw_event = (mode_diff > 0).astype(np.float64)
                        back_event = (mode_diff < 0).astype(np.float64)
                        if np.any(to_fw_event) or np.any(back_event):
                            window = min(31, len(mode_bin))
                            if window % 2 == 0:
                                window -= 1
                            if window < 3:
                                window = 3
                            kernel = np.ones(window, dtype=np.float64)
                            to_fw_est = (np.convolve(to_fw_event, kernel, mode='same') > 0).astype(np.float64)
                            back_est = (np.convolve(back_event, kernel, mode='same') > 0).astype(np.float64)
                            trans_est = ((to_fw_est + back_est) > 0).astype(np.float64)

                            status_df = status_df.with_columns([
                                pl.Series("in_transition_mode_est", trans_est),
                                pl.Series("in_transition_to_fw_est", to_fw_est),
                                pl.Series("in_transition_back_est", back_est),
                            ])
                            self.dataset.topics[status_topic].dataframe = status_df

                            ok_trans = p_spec.render_signal(self.file_name, status_topic, "in_transition_mode_est", color="#4DA3FF")
                            ok_fw = p_spec.render_signal(self.file_name, status_topic, "in_transition_to_fw_est", color="#FF6B6B")
                            ok_back = p_spec.render_signal(self.file_name, status_topic, "in_transition_back_est", color="#33CC66")
                            drew_transition = bool(ok_trans)
                            drew_to_fw = bool(ok_fw)
                            drew_back = bool(ok_back)
                            used_estimated = True

                # Generic fallback: render any transition-like flag columns as binary.
                if not (drew_transition or drew_to_fw or drew_back):
                    for col in status_cols:
                        col_l = col.lower()
                        if "transition" not in col_l:
                            continue
                        if col in (
                            "in_transition_mode_flag", "in_transition_to_fw_flag", "in_transition_back_flag",
                            "in_transition_mode_est", "in_transition_to_fw_est", "in_transition_back_est",
                        ):
                            continue
                        flag_col = f"{col}_flag"
                        status_df, ok = self._add_binary_flag(status_df, col, flag_col)
                        if not ok:
                            continue
                        self.dataset.topics[status_topic].dataframe = status_df
                        color = "#4DA3FF"
                        if "to_fw" in col_l or "forward" in col_l:
                            color = "#FF6B6B"
                        elif "back" in col_l or "to_mc" in col_l or "to_rw" in col_l:
                            color = "#33CC66"
                        rendered = bool(p_spec.render_signal(self.file_name, status_topic, flag_col, color=color))
                        if rendered:
                            if color == "#FF6B6B":
                                drew_to_fw = True
                            elif color == "#33CC66":
                                drew_back = True
                            else:
                                drew_transition = True

                self.dataset.topics[status_topic].dataframe = status_df

            if used_estimated:
                p_spec.plot.setTitle("VTOL: Transition / Back-Transition (Estimated)")
            elif not (drew_transition or drew_to_fw or drew_back or drew_mode):
                p_spec.plot.setTitle("VTOL: Transition Signals Not Found")
                for status_topic in status_topic_candidates:
                    try:
                        cols = self.dataset.topics[status_topic].dataframe.columns
                        print(f"[VTOL Panel] no renderable transition signals in {status_topic}. cols={list(cols)}")
                    except Exception:
                        pass

            # Auto-focus time range to transition interval when current range is too narrow.
            transition_windows = []
            transition_cols = [
                "in_transition_mode_flag", "in_transition_to_fw_flag", "in_transition_back_flag",
                "in_transition_mode_est", "in_transition_to_fw_est", "in_transition_back_est",
                "in_transition_mode", "in_transition_to_fw", "in_transition_back",
            ]
            for status_topic in status_topic_candidates:
                try:
                    df = self.dataset.topics[status_topic].dataframe
                    if "timestamp_sec" not in df.columns:
                        continue
                    t = np.asarray(df["timestamp_sec"].to_numpy(), dtype=np.float64)
                    if len(t) == 0:
                        continue
                    mask = np.zeros(len(t), dtype=bool)
                    for col in transition_cols:
                        if col in df.columns:
                            v = np.asarray(df[col].to_numpy(), dtype=np.float64)
                            if len(v) == len(mask):
                                v = np.where(np.isfinite(v), v, 0.0)
                                mask |= (v > 0.5)
                    if np.any(mask):
                        transition_windows.append((float(np.min(t[mask])), float(np.max(t[mask]))))
                except Exception:
                    continue

            if transition_windows and hasattr(self.workspace, "set_time_range"):
                focus_start = min(w[0] for w in transition_windows)
                focus_end = max(w[1] for w in transition_windows)
                current_duration = float(getattr(self.workspace, "range_end", 0.0) - getattr(self.workspace, "range_start", 0.0))
                transition_duration = max(0.0, focus_end - focus_start)
                # current range가 기본/좁은 경우에만 자동 포커스해서 사용자 수동 범위는 존중
                if current_duration <= 2.0:
                    pad = max(5.0, transition_duration * 0.2)
                    self.workspace.set_time_range(focus_start - pad, focus_end + pad)
            
        else: 
            p_spec.plot.setTitle("MC: Local Z Setpoint vs Actual (Up=Positive)")
            p_spec.render_signal(self.file_name, "vehicle_local_position_setpoint_0", "alt_sp_up", color=COLOR_SP)
            p_spec.render_signal(self.file_name, "vehicle_local_position_0", "alt_up", color=COLOR_ACT)
