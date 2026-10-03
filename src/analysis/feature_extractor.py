import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from .segmenter import FlightSegmenter


class ControlFeatureExtractor:
    """
    Extract response features from setpoint-actual curves.
    MVP scope:
    - multicopter / vtol(MC mode-like) rate loop
    - roll / pitch axes
    """

    AXIS_MAP = {
        "roll": {"sp": "roll", "act": "xyz[0]"},
        "pitch": {"sp": "pitch", "act": "xyz[1]"},
        "yaw": {"sp": "yaw", "act": "xyz[2]"},
    }

    def __init__(self, rules: Optional[dict] = None):
        self.rules = rules or {}
        seg_cfg = (self.rules.get("segmentation", {}) if isinstance(self.rules, dict) else {}) or {}
        self.segmenter = FlightSegmenter(
            hover_hspd_max=float(seg_cfg.get("hover_hspd_max", 2.0)),
            hover_vz_max=float(seg_cfg.get("hover_vz_max", 0.5)),
            climb_vz_min=float(seg_cfg.get("climb_vz_min", 0.8)),
        )

    @staticmethod
    def _with_timestamp_sec(df: pl.DataFrame) -> pl.DataFrame:
        if "timestamp_sec" in df.columns:
            return df
        if "timestamp" in df.columns:
            return df.with_columns((pl.col("timestamp").cast(pl.Float64) / pl.lit(1_000_000.0)).alias("timestamp_sec"))
        return df

    @staticmethod
    def _find_topic(dataset, prefixes) -> Optional[str]:
        if dataset is None or not hasattr(dataset, "topics"):
            return None
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        for p in prefixes:
            if p in dataset.topics:
                return p
        for topic in dataset.topics.keys():
            for p in prefixes:
                if str(topic).startswith(p):
                    return str(topic)
        return None

    @staticmethod
    def _find_signal(df: pl.DataFrame, candidates: List[str]) -> Optional[str]:
        cols = set(df.columns)
        for c in candidates:
            if c in cols:
                return c
        return None

    def _join_ts(self, df_a: pl.DataFrame, sig_a: str, df_b: pl.DataFrame, sig_b: str) -> Optional[pl.DataFrame]:
        df_a = self._with_timestamp_sec(df_a)
        df_b = self._with_timestamp_sec(df_b)
        if "timestamp_sec" not in df_a.columns or "timestamp_sec" not in df_b.columns:
            return None
        try:
            a = (
                df_a.select([pl.col("timestamp_sec").cast(pl.Float64), pl.col(sig_a).cast(pl.Float64).alias("actual")])
                .drop_nulls()
                .sort("timestamp_sec")
            )
            b = (
                df_b.select([pl.col("timestamp_sec").cast(pl.Float64), pl.col(sig_b).cast(pl.Float64).alias("setpoint")])
                .drop_nulls()
                .sort("timestamp_sec")
            )
            if a.height == 0 or b.height == 0:
                return None
            m = a.join_asof(b, on="timestamp_sec", strategy="nearest").drop_nulls(subset=["actual", "setpoint"])
            return m if m.height > 0 else None
        except Exception:
            return None

    @staticmethod
    def _dominant_freq(signal: np.ndarray, dt: float) -> float:
        if signal.size < 16 or dt <= 0:
            return float("nan")
        x = signal - np.nanmean(signal)
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        w = np.hanning(x.size)
        yf = np.fft.rfft(x * w)
        xf = np.fft.rfftfreq(x.size, dt)
        if xf.size <= 1:
            return float("nan")
        amp = np.abs(yf)
        amp[0] = 0.0
        idx = int(np.argmax(amp))
        return float(xf[idx])

    def _extract_actuator_proxy(self, dataset) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        topic = self._find_topic(dataset, ["actuator_outputs", "actuator_motors", "actuator_controls"])
        if not topic:
            return None, None, float("nan")
        try:
            df = self._with_timestamp_sec(dataset.topics[topic].dataframe)
            if "timestamp_sec" not in df.columns:
                return None, None, float("nan")
            cols = [
                c
                for c in df.columns
                if c not in ("timestamp", "timestamp_sec") and ("output" in c.lower() or "control[" in c.lower() or "xyz[" in c.lower())
            ]
            if not cols:
                return None, None, float("nan")
            arr = (
                df.select([pl.col("timestamp_sec").cast(pl.Float64)] + [pl.col(c).cast(pl.Float64).alias(c) for c in cols])
                .drop_nulls()
                .sort("timestamp_sec")
            )
            if arr.height < 4:
                return None, None, float("nan")
            ts = arr["timestamp_sec"].to_numpy()
            mat = arr.select(cols).to_numpy()
            # effort proxy: max absolute normalized channel
            abs_mat = np.nan_to_num(np.abs(mat), nan=0.0, posinf=0.0, neginf=0.0)
            p95 = np.percentile(abs_mat, 95, axis=0)
            scale = np.where(p95 > 1e-6, p95, 1.0)
            norm = abs_mat / scale
            effort = np.nanmax(norm, axis=1)
            # saturation ratio based on high effort events
            sat_ratio = float(np.mean(effort >= 0.98) * 100.0)
            return ts, effort, sat_ratio
        except Exception:
            return None, None, float("nan")

    def _extract_vibration_rms(self, dataset) -> float:
        t = self._find_topic(dataset, ["sensor_combined", "vehicle_acceleration", "vehicle_imu"])
        if not t:
            return float("nan")
        try:
            df = dataset.topics[t].dataframe
            cols = [
                self._find_signal(df, [f"accelerometer_m_s2[{i}]", f"accel_m_s2[{i}]", f"xyz[{i}]", f"accel[{i}]"])
                for i in range(3)
            ]
            rms = []
            for c in cols:
                if not c:
                    continue
                x = np.asarray(df[c].to_numpy(), dtype=np.float64)
                x = x[np.isfinite(x)]
                if x.size == 0:
                    continue
                x = x - np.median(x)
                rms.append(float(np.sqrt(np.mean(x * x))))
            if not rms:
                return float("nan")
            return float(np.nanmax(rms))
        except Exception:
            return float("nan")

    def _extract_innovation_ratio(self, dataset) -> float:
        t = self._find_topic(dataset, ["estimator_status", "ekf2_innovations"])
        if not t:
            return float("nan")
        try:
            df = dataset.topics[t].dataframe
            c = self._find_signal(df, ["innovation_check_flags"])
            if not c:
                return float("nan")
            x = np.asarray(df[c].to_numpy(), dtype=np.float64)
            x = x[np.isfinite(x)]
            if x.size == 0:
                return float("nan")
            return float(np.mean(x > 0) * 100.0)
        except Exception:
            return float("nan")

    @staticmethod
    def _step_windows(t: np.ndarray, sp: np.ndarray, min_step: float, min_gap_s: float, window_s: float) -> List[Tuple[int, int, int]]:
        if t.size < 8 or sp.size < 8:
            return []
        d = np.diff(sp)
        idx = np.where(np.abs(d) >= float(min_step))[0]
        if idx.size == 0:
            return []
        windows = []
        last_t = -1e12
        for i in idx:
            t0 = float(t[i + 1])
            if t0 - last_t < float(min_gap_s):
                continue
            t1 = t0 + float(window_s)
            j0 = int(i + 1)
            j1 = int(np.searchsorted(t, t1, side="right"))
            if j1 - j0 < 8:
                continue
            windows.append((j0, j1, i))
            last_t = t0
        return windows

    def _response_features(self, t: np.ndarray, sp: np.ndarray, y: np.ndarray, cfg_key: str = "response_feature") -> Dict[str, float]:
        cfg = (self.rules.get(cfg_key, {}) if isinstance(self.rules, dict) else {}) or {}
        # Backward-compat alt keys (e.g. attitude_loop section uses *_deg suffix).
        min_step = float(cfg.get("min_step_amplitude", cfg.get("min_step_amplitude_deg", 0.12)))
        min_gap_s = float(cfg.get("min_step_gap_s", 0.6))
        window_s = float(cfg.get("window_s", 2.5))
        settle_tol = float(cfg.get("settling_band_ratio", 0.05))
        rise_ratio = float(cfg.get("rise_target_ratio", 0.9))

        windows = self._step_windows(t, sp, min_step=min_step, min_gap_s=min_gap_s, window_s=window_s)
        if not windows:
            return {
                "maneuver_count": 0.0,
                "overshoot": float("nan"),
                "rise_time": float("nan"),
                "settling_time": float("nan"),
                "steady_state_error": float(np.nanmean(np.abs(y - sp))) if y.size else float("nan"),
                "oscillation_index": float("nan"),
                "dominant_frequency": float("nan"),
            }

        overs, rise_t, settle_t, sse, osc_idx, dom_f = [], [], [], [], [], []
        for j0, j1, i_ref in windows:
            tw = t[j0:j1]
            spw = sp[j0:j1]
            yw = y[j0:j1]
            if tw.size < 8:
                continue
            pre_slice = slice(max(0, j0 - 5), j0)
            y0 = float(np.nanmean(y[pre_slice])) if pre_slice.stop > pre_slice.start else float(yw[0])
            target = float(np.nanmedian(spw[max(0, int(0.7 * spw.size)) :])) if spw.size > 4 else float(spw[-1])
            amp = target - y0
            if not np.isfinite(amp) or abs(amp) < 1e-6:
                continue
            err = yw - target
            n_err = err / abs(amp)
            if amp >= 0:
                ov = float(max(0.0, (np.nanmax(yw) - target) / abs(amp) * 100.0))
                thr = y0 + rise_ratio * amp
                crossed = np.where(yw >= thr)[0]
            else:
                ov = float(max(0.0, (target - np.nanmin(yw)) / abs(amp) * 100.0))
                thr = y0 + rise_ratio * amp
                crossed = np.where(yw <= thr)[0]
            overs.append(ov)

            if crossed.size > 0:
                rise_t.append(float(tw[int(crossed[0])] - tw[0]))

            settle_mask = np.abs(n_err) <= settle_tol
            st = float("nan")
            for k in range(settle_mask.size):
                if bool(settle_mask[k]) and bool(np.all(settle_mask[k:])):
                    st = float(tw[k] - tw[0])
                    break
            settle_t.append(st)

            tail = err[int(0.8 * err.size) :] if err.size > 5 else err
            sse.append(float(np.nanmean(np.abs(tail))))
            osc_idx.append(float(np.nanstd(err) / max(abs(amp), 1e-6) * 100.0))

            dt = float(np.nanmedian(np.diff(tw))) if tw.size > 2 else float("nan")
            dom_f.append(self._dominant_freq(err, dt if np.isfinite(dt) else -1.0))

        def _med(v):
            arr = np.asarray(v, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            return float(np.median(arr)) if arr.size else float("nan")

        return {
            "maneuver_count": float(len(windows)),
            "overshoot": _med(overs),
            "rise_time": _med(rise_t),
            "settling_time": _med(settle_t),
            "steady_state_error": _med(sse),
            "oscillation_index": _med(osc_idx),
            "dominant_frequency": _med(dom_f),
        }

    def extract_rate_loop_features(self, dataset, axes=("roll", "pitch", "yaw")) -> Dict[str, dict]:
        rs_topic = self._find_topic(dataset, "vehicle_rates_setpoint")
        rv_topic = self._find_topic(dataset, "vehicle_angular_velocity")
        if not rs_topic or not rv_topic:
            return {}

        try:
            rs_df = dataset.topics[rs_topic].dataframe
            rv_df = dataset.topics[rv_topic].dataframe
        except Exception:
            return {}

        eff_t, effort, sat_ratio = self._extract_actuator_proxy(dataset)
        vib_rms = self._extract_vibration_rms(dataset)
        innov_ratio = self._extract_innovation_ratio(dataset)

        out = {}
        for axis in axes:
            amap = self.AXIS_MAP.get(axis)
            if not amap:
                continue
            sp_sig = self._find_signal(rs_df, [amap["sp"]])
            act_sig = self._find_signal(rv_df, [amap["act"]])
            if not sp_sig or not act_sig:
                continue
            merged = self._join_ts(rv_df, act_sig, rs_df, sp_sig)
            if merged is None or merged.height < 30:
                continue
            t = merged["timestamp_sec"].to_numpy()
            y = merged["actual"].to_numpy()
            sp = merged["setpoint"].to_numpy()
            mask = np.isfinite(t) & np.isfinite(y) & np.isfinite(sp)
            t = t[mask]
            y = y[mask]
            sp = sp[mask]
            if t.size < 30:
                continue

            resp = self._response_features(t, sp, y)
            tracking_mae = float(np.nanmean(np.abs(y - sp))) if y.size else float("nan")
            # control effort variation from actuator proxy (resampled nearest)
            effort_var = float("nan")
            if eff_t is not None and effort is not None and eff_t.size > 5 and effort.size == eff_t.size:
                idx = np.searchsorted(eff_t, t, side="left")
                idx = np.clip(idx, 0, eff_t.size - 1)
                e = effort[idx]
                de = np.diff(e)
                if de.size > 0:
                    effort_var = float(np.nanstd(de))

            seg_labels = self.segmenter.classify_samples(dataset, t)
            seg_name = "step_like"
            if seg_labels.size:
                uniq, cnt = np.unique(seg_labels, return_counts=True)
                if uniq.size:
                    seg_name = str(uniq[int(np.argmax(cnt))])

            out[axis] = {
                **resp,
                "tracking_mae": tracking_mae,
                "saturation_ratio": float(sat_ratio) if np.isfinite(sat_ratio) else float("nan"),
                "vibration_severity": float(vib_rms) if np.isfinite(vib_rms) else float("nan"),
                "innovation_ratio": float(innov_ratio) if np.isfinite(innov_ratio) else float("nan"),
                "control_effort_variation": effort_var,
                "sample_count": int(t.size),
                "segment": seg_name,
            }
        return out

    ATTITUDE_AXIS_MAP = {
        "roll": ("roll_sp_euler", "roll_euler"),
        "pitch": ("pitch_sp_euler", "pitch_euler"),
        "yaw": ("yaw_sp_euler", "yaw_euler"),
    }

    def extract_attitude_loop_features(self, dataset, axes=("roll", "pitch", "yaw")) -> Dict[str, dict]:
        """Attitude (angle) loop response features from setpoint vs actual Euler.

        Mirrors extract_rate_loop_features but operates on attitude (deg).
        Requires MathEngine.preprocess_dataset() to have populated
        roll_euler/pitch_euler/yaw_euler and roll_sp_euler/... columns.
        """
        sp_topic = self._find_topic(dataset, "vehicle_attitude_setpoint")
        at_topic = self._find_topic(dataset, "vehicle_attitude")
        if not sp_topic or not at_topic:
            return {}
        try:
            sp_df = dataset.topics[sp_topic].dataframe
            at_df = dataset.topics[at_topic].dataframe
        except Exception:
            return {}

        eff_t, effort, sat_ratio = self._extract_actuator_proxy(dataset)
        vib_rms = self._extract_vibration_rms(dataset)
        innov_ratio = self._extract_innovation_ratio(dataset)

        out: Dict[str, dict] = {}
        for axis in axes:
            cols = self.ATTITUDE_AXIS_MAP.get(axis)
            if not cols:
                continue
            sp_col, at_col = cols
            sp_sig = self._find_signal(sp_df, [sp_col])
            at_sig = self._find_signal(at_df, [at_col])
            if not sp_sig or not at_sig:
                continue
            merged = self._join_ts(at_df, at_sig, sp_df, sp_sig)
            if merged is None or merged.height < 30:
                continue
            t = merged["timestamp_sec"].to_numpy()
            y = merged["actual"].to_numpy()
            sp = merged["setpoint"].to_numpy()
            mask = np.isfinite(t) & np.isfinite(y) & np.isfinite(sp)
            t = t[mask]; y = y[mask]; sp = sp[mask]
            if t.size < 30:
                continue

            resp = self._response_features(t, sp, y, cfg_key="attitude_loop")
            attitude_mae = float(np.nanmean(np.abs(y - sp))) if y.size else float("nan")
            out[axis] = {
                **resp,
                "attitude_mae_deg": attitude_mae,
                "saturation_ratio": float(sat_ratio) if np.isfinite(sat_ratio) else float("nan"),
                "vibration_severity": float(vib_rms) if np.isfinite(vib_rms) else float("nan"),
                "innovation_ratio": float(innov_ratio) if np.isfinite(innov_ratio) else float("nan"),
                "sample_count": int(t.size),
                "segment": "attitude_step",
            }
        return out

    RATE_MAX_PARAM_MAP = {
        "roll": "MC_ROLLRATE_MAX",
        "pitch": "MC_PITCHRATE_MAX",
        "yaw": "MC_YAWRATE_MAX",
    }
    # AXIS_MAP keys for rate setpoint columns: vehicle_rates_setpoint.{roll,pitch,yaw}
    RATE_SP_COL_MAP = {"roll": "roll", "pitch": "pitch", "yaw": "yaw"}

    def extract_rate_limit_features(self, dataset, current_params: Optional[dict], axes=("roll", "pitch", "yaw")) -> Dict[str, dict]:
        """% of time |rate_setpoint| approaches MC_*RATE_MAX.

        Returns {} if the rate setpoint topic is missing.
        Per-axis result includes a 'param_present' flag indicating whether the
        MC_*RATE_MAX parameter could be read from the log.
        """
        rs_topic = self._find_topic(dataset, "vehicle_rates_setpoint")
        if not rs_topic:
            return {}
        try:
            rs_df = self._with_timestamp_sec(dataset.topics[rs_topic].dataframe)
        except Exception:
            return {}
        if "timestamp_sec" not in rs_df.columns:
            return {}

        cfg = (self.rules.get("rate_limit", {}) if isinstance(self.rules, dict) else {}) or {}
        near = float(cfg.get("near_threshold_ratio", 0.95))
        params = current_params if isinstance(current_params, dict) else {}

        out: Dict[str, dict] = {}
        for axis in axes:
            sp_col = self.RATE_SP_COL_MAP.get(axis)
            if not sp_col or sp_col not in rs_df.columns:
                continue
            param_name = self.RATE_MAX_PARAM_MAP[axis]
            try:
                max_val = float(params.get(param_name))
            except (TypeError, ValueError):
                max_val = float("nan")
            arr = np.asarray(rs_df[sp_col].to_numpy(), dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size < 10:
                continue
            entry = {
                "rate_max_param": param_name,
                "rate_max_value": max_val if np.isfinite(max_val) else None,
                "sample_count": int(arr.size),
                "param_present": bool(np.isfinite(max_val) and max_val > 1e-6),
            }
            if entry["param_present"]:
                threshold = abs(max_val) * near
                sat_pct = float(np.mean(np.abs(arr) >= threshold) * 100.0)
                entry["saturation_ratio_percent"] = round(sat_pct, 3)
                entry["max_abs_rate_setpoint"] = float(np.nanmax(np.abs(arr)))
            else:
                entry["saturation_ratio_percent"] = None
                entry["max_abs_rate_setpoint"] = float(np.nanmax(np.abs(arr)))
            out[axis] = entry
        return out

    def extract_acc_limit_features(self, dataset, current_params: Optional[dict]) -> dict:
        """Acceleration command saturation against MPC_ACC_*_MAX.

        Returns dict with horizontal_/up_/down_ ratios. param_present flag per axis.
        """
        pos_topic = self._find_topic(dataset, "vehicle_local_position_setpoint")
        if not pos_topic:
            return {}
        try:
            df = self._with_timestamp_sec(dataset.topics[pos_topic].dataframe)
        except Exception:
            return {}
        if "timestamp_sec" not in df.columns:
            return {}

        # Acceleration setpoint columns vary by PX4 version; try common names.
        ax_col = self._find_signal(df, ["acceleration[0]", "acc_x", "ax"])
        ay_col = self._find_signal(df, ["acceleration[1]", "acc_y", "ay"])
        az_col = self._find_signal(df, ["acceleration[2]", "acc_z", "az"])
        if not ax_col or not ay_col or not az_col:
            return {}

        cfg = (self.rules.get("acc_limit", {}) if isinstance(self.rules, dict) else {}) or {}
        near = float(cfg.get("near_threshold_ratio", 0.95))
        params = current_params if isinstance(current_params, dict) else {}

        def _param(name):
            try:
                return float(params.get(name))
            except (TypeError, ValueError):
                return float("nan")

        hor_max = _param("MPC_ACC_HOR_MAX")
        up_max = _param("MPC_ACC_UP_MAX")
        down_max = _param("MPC_ACC_DOWN_MAX")

        ax = np.asarray(df[ax_col].to_numpy(), dtype=np.float64)
        ay = np.asarray(df[ay_col].to_numpy(), dtype=np.float64)
        az = np.asarray(df[az_col].to_numpy(), dtype=np.float64)
        n = min(ax.size, ay.size, az.size)
        if n < 10:
            return {}
        ax, ay, az = ax[:n], ay[:n], az[:n]
        mask = np.isfinite(ax) & np.isfinite(ay) & np.isfinite(az)
        ax, ay, az = ax[mask], ay[mask], az[mask]
        if ax.size < 10:
            return {}

        hor = np.hypot(ax, ay)
        # NED: az>0 is descent (down), az<0 is ascent (up).
        up_cmd = np.where(az < 0, -az, 0.0)
        down_cmd = np.where(az > 0, az, 0.0)

        def _sat(cmd, limit):
            if not (np.isfinite(limit) and limit > 1e-6):
                return None
            return round(float(np.mean(cmd >= (abs(limit) * near)) * 100.0), 3)

        return {
            "horizontal": {
                "param": "MPC_ACC_HOR_MAX",
                "limit": hor_max if np.isfinite(hor_max) else None,
                "saturation_ratio_percent": _sat(hor, hor_max),
                "param_present": bool(np.isfinite(hor_max) and hor_max > 1e-6),
                "max_cmd": float(np.nanmax(hor)),
            },
            "up": {
                "param": "MPC_ACC_UP_MAX",
                "limit": up_max if np.isfinite(up_max) else None,
                "saturation_ratio_percent": _sat(up_cmd, up_max),
                "param_present": bool(np.isfinite(up_max) and up_max > 1e-6),
                "max_cmd": float(np.nanmax(up_cmd)),
            },
            "down": {
                "param": "MPC_ACC_DOWN_MAX",
                "limit": down_max if np.isfinite(down_max) else None,
                "saturation_ratio_percent": _sat(down_cmd, down_max),
                "param_present": bool(np.isfinite(down_max) and down_max > 1e-6),
                "max_cmd": float(np.nanmax(down_cmd)),
            },
            "sample_count": int(ax.size),
        }

    def extract_tilt_limit_features(self, dataset, current_params: Optional[dict]) -> dict:
        """Setpoint tilt vs MPC_TILTMAX_AIR.

        Tilt computed from roll_sp_euler/pitch_sp_euler:
            tilt_deg = degrees(arccos(cos(roll) * cos(pitch)))
        """
        sp_topic = self._find_topic(dataset, "vehicle_attitude_setpoint")
        if not sp_topic:
            return {}
        try:
            df = self._with_timestamp_sec(dataset.topics[sp_topic].dataframe)
        except Exception:
            return {}
        r_col = self._find_signal(df, ["roll_sp_euler"])
        p_col = self._find_signal(df, ["pitch_sp_euler"])
        if not r_col or not p_col:
            return {}

        cfg = (self.rules.get("tilt_limit", {}) if isinstance(self.rules, dict) else {}) or {}
        near = float(cfg.get("near_threshold_ratio", 0.95))
        params = current_params if isinstance(current_params, dict) else {}
        try:
            tilt_max_deg = float(params.get("MPC_TILTMAX_AIR"))
        except (TypeError, ValueError):
            tilt_max_deg = float("nan")

        r = np.deg2rad(np.asarray(df[r_col].to_numpy(), dtype=np.float64))
        p = np.deg2rad(np.asarray(df[p_col].to_numpy(), dtype=np.float64))
        n = min(r.size, p.size)
        if n < 10:
            return {}
        r, p = r[:n], p[:n]
        mask = np.isfinite(r) & np.isfinite(p)
        r, p = r[mask], p[mask]
        if r.size < 10:
            return {}
        cos_tilt = np.cos(r) * np.cos(p)
        tilt_deg = np.degrees(np.arccos(np.clip(cos_tilt, -1.0, 1.0)))

        param_present = bool(np.isfinite(tilt_max_deg) and tilt_max_deg > 1e-6)
        sat_pct = None
        if param_present:
            sat_pct = round(float(np.mean(tilt_deg >= (abs(tilt_max_deg) * near)) * 100.0), 3)

        return {
            "param": "MPC_TILTMAX_AIR",
            "limit_deg": tilt_max_deg if np.isfinite(tilt_max_deg) else None,
            "saturation_ratio_percent": sat_pct,
            "param_present": param_present,
            "max_tilt_deg": float(np.nanmax(tilt_deg)),
            "sample_count": int(r.size),
        }

    def extract_tecs_altitude_features(self, dataset) -> dict:
        """TECS altitude tracking error.

        Uses MathEngine-provided alt_up / alt_sp_up columns (NED Z converted to
        Up). Filters out samples where vehicle isn't flying forward (ground
        speed proxy) if available.
        """
        pos_topic = self._find_topic(dataset, "vehicle_local_position")
        sp_topic = self._find_topic(dataset, "vehicle_local_position_setpoint")
        if not pos_topic or not sp_topic:
            return {}
        try:
            pos_df = dataset.topics[pos_topic].dataframe
            sp_df = dataset.topics[sp_topic].dataframe
        except Exception:
            return {}

        alt_col = self._find_signal(pos_df, ["alt_up"])
        alt_sp_col = self._find_signal(sp_df, ["alt_sp_up"])
        if not alt_col or not alt_sp_col:
            return {}

        merged = self._join_ts(pos_df, alt_col, sp_df, alt_sp_col)
        if merged is None or merged.height < 30:
            return {}
        t = merged["timestamp_sec"].to_numpy()
        y = merged["actual"].to_numpy()
        sp = merged["setpoint"].to_numpy()
        mask = np.isfinite(t) & np.isfinite(y) & np.isfinite(sp)
        t = t[mask]; y = y[mask]; sp = sp[mask]
        if t.size < 30:
            return {}

        # Forward-flight gating: keep only samples where horizontal ground speed
        # is > 5 m/s. Falls back to "all samples" when ground_speed_mag absent.
        try:
            gs_col = self._find_signal(pos_df, ["ground_speed_mag"])
            if gs_col:
                pos_df_ts = self._with_timestamp_sec(pos_df)
                gs_t = pos_df_ts["timestamp_sec"].to_numpy()
                gs = pos_df_ts[gs_col].to_numpy()
                if gs_t.size > 0:
                    idx = np.searchsorted(gs_t, t, side="left")
                    idx = np.clip(idx, 0, gs_t.size - 1)
                    fwd_mask = np.isfinite(gs[idx]) & (gs[idx] > 5.0)
                    if int(np.count_nonzero(fwd_mask)) >= 30:
                        t = t[fwd_mask]; y = y[fwd_mask]; sp = sp[fwd_mask]
        except Exception:
            pass

        err = y - sp
        mae = float(np.nanmean(np.abs(err)))
        max_abs_err = float(np.nanmax(np.abs(err)))
        return {
            "altitude_mae_m": round(mae, 3),
            "altitude_max_abs_error_m": round(max_abs_err, 3),
            "sample_count": int(t.size),
        }

    def extract_tecs_airspeed_features(self, dataset) -> dict:
        """TECS airspeed tracking error.

        Prefers tecs_status topic when present (true_airspeed_sp vs
        true_airspeed_filtered). Falls back to airspeed.true_airspeed_m_s vs
        vehicle_local_position ground-speed magnitude (approximate, no wind
        correction).
        """
        tecs_topic = self._find_topic(dataset, "tecs_status")
        if tecs_topic:
            try:
                df = self._with_timestamp_sec(dataset.topics[tecs_topic].dataframe)
            except Exception:
                df = None
            if df is not None and "timestamp_sec" in df.columns:
                sp_col = self._find_signal(df, ["true_airspeed_sp", "tas_sp", "airspeed_sp"])
                act_col = self._find_signal(df, ["true_airspeed_filtered", "tas_filtered", "true_airspeed"])
                if sp_col and act_col:
                    sp_arr = np.asarray(df[sp_col].to_numpy(), dtype=np.float64)
                    act_arr = np.asarray(df[act_col].to_numpy(), dtype=np.float64)
                    n = min(sp_arr.size, act_arr.size)
                    if n >= 30:
                        sp_arr, act_arr = sp_arr[:n], act_arr[:n]
                        mask = np.isfinite(sp_arr) & np.isfinite(act_arr)
                        sp_arr, act_arr = sp_arr[mask], act_arr[mask]
                        if sp_arr.size >= 30:
                            err = act_arr - sp_arr
                            return {
                                "airspeed_mae_mps": round(float(np.nanmean(np.abs(err))), 3),
                                "airspeed_max_abs_error_mps": round(float(np.nanmax(np.abs(err))), 3),
                                "sample_count": int(sp_arr.size),
                                "source": "tecs_status",
                            }

        # Fallback: airspeed vs ground speed (no wind correction).
        air_topic = self._find_topic(dataset, "airspeed")
        pos_topic = self._find_topic(dataset, "vehicle_local_position")
        if not air_topic or not pos_topic:
            return {}
        try:
            air_df = self._with_timestamp_sec(dataset.topics[air_topic].dataframe)
            pos_df = self._with_timestamp_sec(dataset.topics[pos_topic].dataframe)
        except Exception:
            return {}
        air_col = self._find_signal(air_df, ["true_airspeed_m_s", "indicated_airspeed_m_s"])
        gs_col = self._find_signal(pos_df, ["ground_speed_mag"])
        if not air_col or not gs_col:
            return {}
        merged = self._join_ts(air_df, air_col, pos_df, gs_col)
        if merged is None or merged.height < 30:
            return {}
        air = merged["actual"].to_numpy()
        gs = merged["setpoint"].to_numpy()
        mask = np.isfinite(air) & np.isfinite(gs) & (gs > 5.0)
        air = air[mask]; gs = gs[mask]
        if air.size < 30:
            return {}
        residual = air - gs
        return {
            "airspeed_mae_mps": None,
            "airspeed_residual_rms_mps": round(float(np.sqrt(np.mean(residual ** 2))), 3),
            "sample_count": int(air.size),
            "source": "airspeed_vs_groundspeed_fallback",
        }
