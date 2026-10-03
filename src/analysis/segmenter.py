import numpy as np
import polars as pl


class FlightSegmenter:
    """
    Lightweight segment helper for tuning analysis.
    Current MVP supports coarse hover/climb tagging from local velocity.
    """

    def __init__(self, hover_hspd_max: float = 2.0, hover_vz_max: float = 0.5, climb_vz_min: float = 0.8):
        self.hover_hspd_max = float(hover_hspd_max)
        self.hover_vz_max = float(hover_vz_max)
        self.climb_vz_min = float(climb_vz_min)

    @staticmethod
    def _with_timestamp_sec(df: pl.DataFrame) -> pl.DataFrame:
        if "timestamp_sec" in df.columns:
            return df
        if "timestamp" in df.columns:
            return df.with_columns((pl.col("timestamp").cast(pl.Float64) / pl.lit(1_000_000.0)).alias("timestamp_sec"))
        return df

    @staticmethod
    def _find_signal(df: pl.DataFrame, candidates):
        cols = set(df.columns)
        for c in candidates:
            if c in cols:
                return c
        return None

    def classify_samples(self, dataset, time_series: np.ndarray):
        """
        Return per-sample segment labels aligned to input timestamps.
        Labels: hover / climb / maneuver / unknown
        """
        if time_series is None or len(time_series) == 0:
            return np.array([], dtype=object)
        if dataset is None or not hasattr(dataset, "topics"):
            return np.array(["unknown"] * len(time_series), dtype=object)

        topic_name = None
        for k in dataset.topics.keys():
            if str(k).startswith("vehicle_local_position"):
                topic_name = k
                break
        if not topic_name:
            return np.array(["unknown"] * len(time_series), dtype=object)

        try:
            df = self._with_timestamp_sec(dataset.topics[topic_name].dataframe)
            if "timestamp_sec" not in df.columns:
                return np.array(["unknown"] * len(time_series), dtype=object)
            vx = self._find_signal(df, ["vx", "vel_x"])
            vy = self._find_signal(df, ["vy", "vel_y"])
            vz = self._find_signal(df, ["vz", "vel_z"])
            if not vx or not vy or not vz:
                return np.array(["unknown"] * len(time_series), dtype=object)

            pos = (
                df.select(
                    [
                        pl.col("timestamp_sec").cast(pl.Float64),
                        pl.col(vx).cast(pl.Float64).alias("vx"),
                        pl.col(vy).cast(pl.Float64).alias("vy"),
                        pl.col(vz).cast(pl.Float64).alias("vz"),
                    ]
                )
                .drop_nulls()
                .sort("timestamp_sec")
            )
            if pos.height <= 0:
                return np.array(["unknown"] * len(time_series), dtype=object)

            ts_df = pl.DataFrame({"timestamp_sec": np.asarray(time_series, dtype=np.float64)}).sort("timestamp_sec")
            merged = ts_df.join_asof(pos, on="timestamp_sec", strategy="nearest").drop_nulls(subset=["vx", "vy", "vz"])
            if merged.height <= 0:
                return np.array(["unknown"] * len(time_series), dtype=object)

            hspd = np.sqrt(merged["vx"].to_numpy() ** 2 + merged["vy"].to_numpy() ** 2)
            vz_abs = np.abs(merged["vz"].to_numpy())
            labels = np.full(merged.height, "maneuver", dtype=object)
            hover_mask = (hspd <= self.hover_hspd_max) & (vz_abs <= self.hover_vz_max)
            labels[hover_mask] = "hover"
            climb_mask = (~hover_mask) & (vz_abs >= self.climb_vz_min)
            labels[climb_mask] = "climb"

            # Map back to original ordering
            out = np.array(["unknown"] * len(time_series), dtype=object)
            idx_sort = np.argsort(np.asarray(time_series, dtype=np.float64))
            out[idx_sort[: labels.shape[0]]] = labels
            return out
        except Exception:
            return np.array(["unknown"] * len(time_series), dtype=object)

