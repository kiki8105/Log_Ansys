"""Generate tiny structural fixtures for the multi-format acceptance gate.

These files only exercise parser and workspace contracts.  The generated
manifest marks every generated source as ``synthetic`` so
``validate_format_acceptance.py`` can never treat it as release evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_ros1(path: Path) -> None:
    from rosbags.rosbag1 import Writer
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)
    Float64 = typestore.types["std_msgs/msg/Float64"]
    with Writer(path) as writer:
        connection = writer.add_connection(
            "/acceptance/value", Float64.__msgtype__, typestore=typestore
        )
        for index, value in enumerate((1.0, 2.5, -0.5)):
            message = Float64(data=value)
            writer.write(
                connection,
                1_000_000_000 + index * 100_000_000,
                typestore.serialize_ros1(message, message.__msgtype__),
            )


def _write_ros2(directory: Path, *, mcap: bool) -> Path:
    from rosbags.rosbag2 import StoragePlugin, Writer
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.LATEST)
    Float64 = typestore.types["std_msgs/msg/Float64"]
    kwargs = {"version": 9}
    if mcap:
        kwargs["storage_plugin"] = StoragePlugin.MCAP
    with Writer(directory, **kwargs) as writer:
        connection = writer.add_connection(
            "/acceptance/value", Float64.__msgtype__, typestore=typestore
        )
        for index, value in enumerate((3.0, 4.5, 2.0)):
            message = Float64(data=value)
            writer.write(
                connection,
                2_000_000_000 + index * 200_000_000,
                typestore.serialize_cdr(message, message.__msgtype__),
            )
    suffix = "*.mcap" if mcap else "*.db3"
    return next(directory.glob(suffix))


def _fmt_packet(message_type: int, message_length: int) -> bytes:
    return (
        b"\xA3\x95\x80"
        + bytes((message_type, message_length))
        + b"IMU\x00"
        + b"Qfff".ljust(16, b"\x00")
        + b"TimeUS,GyrX,GyrY,GyrZ".ljust(64, b"\x00")
    )


def _write_dataflash_binary(path: Path) -> None:
    message_type = 129
    row_format = "<Qfff"
    message_length = 3 + struct.calcsize(row_format)
    payload = bytearray(_fmt_packet(message_type, message_length))
    for timestamp_us, x, y, z in (
        (1_000_000, 0.1, 0.2, 0.3),
        (1_100_000, 0.2, 0.3, 0.4),
        (1_200_000, 0.3, 0.4, 0.5),
    ):
        payload.extend(b"\xA3\x95" + bytes((message_type,)))
        payload.extend(struct.pack(row_format, timestamp_us, x, y, z))
    path.write_bytes(payload)


def _write_dataflash_text(path: Path) -> None:
    path.write_text(
        "FMT, 128, 89, FMT, BBnNZ, Type,Length,Name,Format,Columns\n"
        "FMT, 129, 23, IMU, Qfff, TimeUS,GyrX,GyrY,GyrZ\n"
        "IMU, 1000000, 0.1, 0.2, 0.3\n"
        "IMU, 1100000, 0.2, 0.3, 0.4\n"
        "IMU, 1200000, 0.3, 0.4, 0.5\n",
        encoding="ascii",
    )


def _write_tlog(path: Path) -> None:
    from pymavlink.dialects.v20 import ardupilotmega

    mav = ardupilotmega.MAVLink(None)
    mav.srcSystem = 1
    mav.srcComponent = 1
    payload = bytearray()
    for index, roll in enumerate((0.1, 0.2, -0.1)):
        timestamp_us = 1_700_000_000_000_000 + index * 100_000
        message = ardupilotmega.MAVLink_attitude_message(
            time_boot_ms=1000 + index * 100,
            roll=roll,
            pitch=0.05 * index,
            yaw=0.2 * index,
            rollspeed=0.01,
            pitchspeed=0.02,
            yawspeed=0.03,
        )
        payload.extend(struct.pack(">Q", timestamp_us))
        payload.extend(message.pack(mav))
    path.write_bytes(payload)


def generate(output_dir: Path, ulg_path: Path | None = None) -> Path:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(
            f"synthetic fixture output must be a new or empty directory: {output_dir}"
        )

    ros1 = output_dir / "synthetic_ros1.bag"
    ros2_db3_dir = output_dir / "synthetic_ros2_db3"
    ros2_mcap_dir = output_dir / "synthetic_ros2_mcap"
    dataflash_bin = output_dir / "synthetic_dataflash.bin"
    dataflash_log = output_dir / "synthetic_dataflash.log"
    tlog = output_dir / "synthetic_telemetry.tlog"
    csv_path = output_dir / "synthetic_table.csv"
    json_path = output_dir / "synthetic_table.json"

    _write_ros1(ros1)
    ros2_db3 = _write_ros2(ros2_db3_dir, mcap=False)
    ros2_mcap = _write_ros2(ros2_mcap_dir, mcap=True)
    _write_dataflash_binary(dataflash_bin)
    _write_dataflash_text(dataflash_log)
    _write_tlog(tlog)
    csv_path.write_text(
        "timestamp_sec,value\n0.0,1.0\n0.1,2.0\n0.2,3.0\n",
        encoding="utf-8",
    )
    json_path.write_text(
        json.dumps(
            [
                {"timestamp_sec": 0.0, "value": 1.0},
                {"timestamp_sec": 0.1, "value": 2.0},
                {"timestamp_sec": 0.2, "value": 3.0},
            ]
        ),
        encoding="utf-8",
    )

    fixtures = {
        "ros1_bag": {"path": str(ros1), "provenance": "synthetic"},
        "ros2_db3": {"path": str(ros2_db3), "provenance": "synthetic"},
        "ros2_mcap": {"path": str(ros2_mcap), "provenance": "synthetic"},
        "ardupilot_bin": {"path": str(dataflash_bin), "provenance": "synthetic"},
        "ardupilot_log": {"path": str(dataflash_log), "provenance": "synthetic"},
        "mavlink_tlog": {"path": str(tlog), "provenance": "synthetic"},
        "csv": {"path": str(csv_path), "provenance": "synthetic"},
        "json": {"path": str(json_path), "provenance": "synthetic"},
    }
    if ulg_path is not None:
        fixtures["px4_ulog"] = {
            "path": str(ulg_path.expanduser().resolve()),
            "provenance": "real",
            "notes": "User-supplied real ULog; all generated siblings remain synthetic.",
        }
    manifest = output_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"schema_version": 1, "fixtures": fixtures},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runtime" / "synthetic-format-fixtures",
    )
    parser.add_argument(
        "--ulg",
        type=Path,
        help="Optional real ULog to reference as the only real fixture in the manifest.",
    )
    args = parser.parse_args()
    manifest = generate(args.output_dir, args.ulg)
    print(f"MANIFEST={manifest}")
    print("Synthetic fixtures are regression-only and cannot release a format.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
