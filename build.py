from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


APP_NAME = "Log ansys"
SPEC_NAME = "Log_ansys.spec"

REQUIRED_IMPORTS = {
    "PyInstaller": "PyInstaller",
    "PySide6": "PySide6",
    "pyqtgraph": "pyqtgraph",
    "numpy": "numpy",
    "polars": "polars",
    "scipy": "scipy",
    "pyulog": "pyulog",
    "rosbags": "rosbags",
    "pymavlink": "pymavlink",
    "yaml": "PyYAML",
}

OPTIONAL_IMPORTS = {
    "pandas": "pandas",
    "pyarrow": "pyarrow",
}

ICON_CANDIDATES = (
    Path("assets") / "icons" / "custom_app.ico",
)

QT_MSVC_RUNTIME_FILES = (
    "MSVCP140.dll",
    "MSVCP140_1.dll",
    "MSVCP140_2.dll",
    "VCRUNTIME140.dll",
    "VCRUNTIME140_1.dll",
)


def project_root() -> Path:
    return Path(__file__).resolve().parent


def workspace_root() -> Path:
    return project_root().parent


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def check_dependencies(skip_optional: bool = False) -> None:
    missing = [pip_name for module, pip_name in REQUIRED_IMPORTS.items() if not module_available(module)]
    if missing:
        packages = " ".join(missing)
        raise SystemExit(
            "\n[ERROR] EXE build dependencies are missing.\n"
            f"        Missing packages: {packages}\n"
            "        Run this first:\n"
            "        python -m pip install -r requirements-build.txt\n"
        )

    if skip_optional:
        return

    optional_missing = [pip_name for module, pip_name in OPTIONAL_IMPORTS.items() if not module_available(module)]
    if optional_missing:
        print(
            "[WARN] Optional packages are not installed: "
            + ", ".join(optional_missing)
            + "\n       Current build does not require them, but some future export/debug paths may."
        )


def find_default_icon() -> Path | None:
    # (2026-10-03) 빌드 입력은 저장소 내부 자산만 사용해 부모 폴더 배치에 의존하지 않습니다.
    for name in ICON_CANDIDATES:
        candidate = project_root() / name
        if candidate.exists():
            return candidate
    return None


SPLASH_REL_PATH = Path("assets") / "splash" / "custom_splash.png"


def splash_image_path() -> Path:
    return project_root() / SPLASH_REL_PATH


def check_splash_image() -> None:
    """Fail fast if the bootloader splash image is missing — PyInstaller
    would otherwise abort deep in the spec processing with a less clear
    message. (2026-05-20)"""
    path = splash_image_path()
    if not path.is_file():
        raise SystemExit(
            "\n[ERROR] Bootloader splash image is missing.\n"
            f"        Expected at: {path}\n"
            "        Add your own redistributable image at that path or build without --with-splash.\n"
        )


def bootloader_splash_supported() -> bool:
    """Return whether this build environment can use PyInstaller Splash.

    PyInstaller's splash feature depends on tkinter/Tcl-Tk support in the
    builder's Python environment. Some Windows Python installs omit it, in
    which case we should fall back to a normal build instead of aborting.
    """
    return module_available("tkinter")


def prepare_splash_image() -> Path:
    """Return a PNG version of the splash image with an RGB color mode.

    PyInstaller's `Splash(...)` rasterizes the source image into PNG when
    embedding into the bootloader; PNG cannot represent CMYK, so JPGs
    exported from a print-oriented editor (Photoshop default for some
    presets) fail with 'cannot write mode CMYK as PNG'. We pre-convert
    to RGB and cache the result next to the source so repeat builds
    skip the work. (2026-05-20)"""
    src = splash_image_path()
    check_splash_image()
    cache_path = src.with_name(src.stem + ".rgb.png")
    try:
        src_mtime = src.stat().st_mtime
    except Exception:
        src_mtime = 0.0
    if cache_path.is_file():
        try:
            if cache_path.stat().st_mtime >= src_mtime:
                return cache_path
        except Exception:
            pass
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "\n[ERROR] Pillow is required to prepare the splash image.\n"
            f"        Reason: {exc}\n"
            "        Install with: python -m pip install Pillow\n"
        ) from exc
    try:
        with Image.open(src) as img:
            converted = img.convert("RGB") if img.mode != "RGB" else img.copy()
            converted.save(cache_path, format="PNG")
    except Exception as exc:
        raise SystemExit(
            "\n[ERROR] Failed to prepare splash image (CMYK → RGB conversion).\n"
            f"        Source: {src}\n"
            f"        Reason: {exc}\n"
        ) from exc
    print(f"[OK] Splash image normalized to RGB PNG: {cache_path}")
    return cache_path


def spec_text(console: bool, onefile: bool, splash_image: Path | None = None) -> str:
    console_literal = "True" if console else "False"
    bundle_mode = "onefile" if onefile else "onedir"
    icon_path = find_default_icon()
    if icon_path:
        icon_rel_path = icon_path.relative_to(project_root()).as_posix()
        icon_literal = f"str(project_root / {icon_rel_path!r})"
    else:
        icon_literal = "None"
    splash_enabled = splash_image is not None

    # 2026-05-20: bootloader-level splash (Tcl/Tk based) so the image is
    # visible during the multi-second Python/PySide6 import phase that
    # precedes QApplication. Closed by main.py via `pyi_splash.close()`
    # once the main window is shown. Single fixed image — random rotation
    # remains available only in the dev (`python main.py`) QSplashScreen
    # fallback path.
    # The path is baked in as an absolute literal so the spec works
    # regardless of cwd. Source image is normalized to RGB PNG upstream
    # (build.py `prepare_splash_image`) because PyInstaller embeds the
    # image as PNG and PNG cannot represent CMYK JPEGs.
    splash_literal = repr(str(splash_image.resolve())) if splash_image else "None"
    splash_block = f"""
splash_image = {splash_literal}
splash = Splash(
    splash_image,
    binaries=a.binaries,
    datas=a.datas,
    text_pos=None,
    text_size=12,
    minify_script=True,
    always_on_top=True,
)
"""
    if not splash_enabled:
        splash_block = ""

    # (2026-05-28) UPX 비활성화: Windows Defender 등 백신이 UPX 패킹 exe 를 false positive
    # 로 잡는 사례가 흔함. 사이즈 ~30% 줄지만 신뢰성·배포 마찰 줄이는 게 우선.
    if onefile:
        splash_args = "splash,\n    splash.binaries,\n    " if splash_enabled else ""
        build_tail = f"""{splash_block}
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    {splash_args}[],
    name={APP_NAME!r},
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console={console_literal},
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_path,
)
"""
    else:
        splash_args = "splash,\n    " if splash_enabled else ""
        collect_splash_args = "splash.binaries,\n    " if splash_enabled else ""
        build_tail = f"""{splash_block}
exe = EXE(
    pyz,
    a.scripts,
    {splash_args}[],
    exclude_binaries=True,
    name={APP_NAME!r},
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console={console_literal},
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    contents_directory=".",
    icon=icon_path,
)

coll = COLLECT(
    exe,
    {collect_splash_args}a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name={APP_NAME!r},
)
"""

    return f'''# -*- mode: python ; coding: utf-8 -*-
#
# Auto-generated by build.py
# Bundle mode: {bundle_mode}

from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_submodules, copy_metadata
import PySide6

block_cipher = None

project_root = Path(SPECPATH).resolve()
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))


def existing_data(rel_path: str, dest: str):
    path = (project_root / rel_path).resolve()
    if not path.exists():
        return []
    return [(str(path), dest)]


icon_path = {icon_literal}

# Qt 6.11 wheels are built against a newer MSVC runtime than some Python
# distributions bundle.  Supplying the wheel's matching runtime at the
# executable root prevents Windows from resolving an older VCRUNTIME first.
pyside_root = Path(PySide6.__file__).resolve().parent
qt_msvc_runtime_names = {QT_MSVC_RUNTIME_FILES!r}
qt_msvc_binaries = [
    (str(pyside_root / name), ".")
    for name in qt_msvc_runtime_names
    if (pyside_root / name).is_file()
]

datas = []
datas += existing_data("assets", "assets")
datas += existing_data("config", "config")
datas += existing_data("README.md", ".")

for package_name in ("PySide6", "pyqtgraph", "numpy", "scipy", "polars", "pyulog", "rosbags", "pymavlink", "PyYAML"):
    try:
        datas += copy_metadata(package_name)
    except Exception:
        pass

hiddenimports = []
for module_name in (
    "analysis",
    "core",
    "engines",
    "gui",
    "storage",
    "visualization",
    "readers",
    "pyulog",
    "rosbags",
):
    try:
        hiddenimports += collect_submodules(module_name)
    except Exception:
        pass

hiddenimports += [
    "pyqtgraph.exporters",
    "pyqtgraph.exporters.ImageExporter",
    "scipy.interpolate",
    "scipy.signal",
    "yaml",
    "pymavlink.DFReader",
    "pymavlink.mavutil",
    "pymavlink.dialects.v10.ardupilotmega",
    "pymavlink.dialects.v20.ardupilotmega",
]
hiddenimports = sorted(set(hiddenimports))

excludes = [
    "IPython",
    "jupyter",
    "jupyter_client",
    "jupyter_core",
    "matplotlib",
    "matplotlib.tests",
    "nbformat",
    "numpy.tests",
    "pandas",
    "pandas.tests",
    "pytest",
    "tests",
    "tkinter",
    "zmq",
]

a = Analysis(
    [str(src_dir / "main.py")],
    pathex=[str(src_dir), str(project_root)],
    binaries=qt_msvc_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
{build_tail}
'''


def write_spec(root: Path, console: bool, onefile: bool, splash_image: Path | None = None) -> Path:
    spec_path = root / SPEC_NAME
    spec_path.write_text(
        spec_text(console=console, onefile=onefile, splash_image=splash_image),
        encoding="utf-8",
    )
    print(f"[OK] Spec file written: {spec_path}")
    return spec_path


def verify_build(root: Path, onefile: bool) -> None:
    if onefile:
        exe_path = root / "dist" / f"{APP_NAME}.exe"
        if not exe_path.exists():
            raise SystemExit(f"[ERROR] Built EXE not found: {exe_path}")
        print(f"[OK] One-file EXE created: {exe_path}")
        print("[INFO] This EXE is intended for single-file distribution.")
        return

    dist_dir = root / "dist" / APP_NAME
    exe_path = dist_dir / f"{APP_NAME}.exe"
    if not exe_path.exists():
        raise SystemExit(f"[ERROR] Built EXE not found: {exe_path}")

    required_runtime_files = [
        dist_dir / "config" / "thresholds.yaml",
        dist_dir / "config" / "advisor_registry.yaml",
        dist_dir / "config" / "parameter_recommendations.yaml",
    ]
    missing_runtime = [str(path) for path in required_runtime_files if not path.exists()]
    if missing_runtime:
        raise SystemExit(
            "[ERROR] Required runtime config files are missing from the onedir bundle.\n"
            + "\n".join(f"        - {path}" for path in missing_runtime)
        )

    print(f"[OK] Onedir EXE created: {exe_path}")
    print("[OK] Required config files included in bundle.")


def sanitized_pyinstaller_environment(
    base_environment: dict[str, str] | None = None,
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Remove host-tool native DLL folders from PyInstaller's search path.

    Codex Desktop exposes Poppler/libheif runtimes on ``PATH`` for document
    tooling.  PyInstaller may otherwise collect their ICU/UCRT DLLs into an
    unrelated application.  A foreign ``icuuc.dll`` at the EXE root shadows
    Windows' ICU and makes ``PySide6.QtCore`` fail before application startup.
    """
    environment = dict(os.environ if base_environment is None else base_environment)
    raw_path = environment.get("PATH", "")
    kept: list[str] = []
    removed: list[str] = []
    for entry in raw_path.split(os.pathsep):
        normalized = entry.replace("/", "\\").casefold()
        is_host_native_runtime = (
            "\\.cache\\codex-runtimes\\" in normalized
            and "\\dependencies\\native\\" in normalized
        )
        if is_host_native_runtime:
            removed.append(entry)
        else:
            kept.append(entry)
    environment["PATH"] = os.pathsep.join(kept)
    return environment, tuple(removed)


def run_pyinstaller(root: Path, spec_path: Path, clean: bool) -> None:
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm"]
    if clean:
        cmd.append("--clean")
    cmd.append(str(spec_path))

    environment, removed_paths = sanitized_pyinstaller_environment()
    if removed_paths:
        print(
            "[INFO] Isolated PyInstaller from host-tool native DLL paths: "
            f"{len(removed_paths)} PATH entr{'y' if len(removed_paths) == 1 else 'ies'} removed."
        )
    print("[RUN] " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(root), check=True, env=environment)


def normalize_onedir_qt_runtime(root: Path, onefile: bool) -> None:
    """Place the PySide6-matched MSVC runtime beside the onedir executable."""
    if onefile:
        return
    try:
        import PySide6
    except ImportError as exc:
        raise SystemExit(f"[ERROR] Cannot locate PySide6 runtime files: {exc}") from exc

    dist_dir = root / "dist" / APP_NAME
    pyside_root = Path(PySide6.__file__).resolve().parent
    copied: list[str] = []
    for name in QT_MSVC_RUNTIME_FILES:
        source = pyside_root / name
        if not source.is_file():
            continue
        shutil.copy2(source, dist_dir / name)
        copied.append(name)
    if not copied:
        raise SystemExit("[ERROR] No PySide6 MSVC runtime files were available for the onedir bundle.")
    print(f"[OK] Qt-matched MSVC runtime normalized: {', '.join(copied)}")


def verify_no_host_runtime_contamination(root: Path) -> None:
    """Reject builds that captured DLLs from Codex/Desktop helper runtimes."""
    analysis_toc = root / "build" / Path(SPEC_NAME).stem / "Analysis-00.toc"
    if not analysis_toc.is_file():
        return
    content = analysis_toc.read_text(encoding="utf-8", errors="replace").replace("/", "\\").casefold()
    if "\\.cache\\codex-runtimes\\" in content and "\\dependencies\\native\\" in content:
        raise SystemExit(
            "[ERROR] Build captured native DLLs from the host tooling runtime. "
            "Do not distribute this bundle; check PATH isolation."
        )
    print("[OK] No host-tool native DLL contamination detected.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Build the Windows EXE for {APP_NAME}",
    )
    parser.add_argument("--clean", action="store_true", help="Remove PyInstaller cache and rebuild from scratch.")
    parser.add_argument("--console", action="store_true", help="Build a console-enabled EXE for debugging.")
    parser.add_argument("--write-spec-only", action="store_true", help="Only regenerate the .spec file.")
    parser.add_argument("--skip-optional-check", action="store_true", help="Skip pandas/pyarrow warning output.")
    # (2026-05-28) Default = onedir (폴더 통째 — 첫 실행 빠르고, temp dir 경고 / 백신 false positive
    # 회피). 휴대는 dist 폴더를 그대로 USB 에 넣거나 --zip 으로 자동 zip 생성.
    parser.add_argument("--onedir", action="store_true", help="(deprecated, default 동작) onedir 번들 생성.")
    parser.add_argument("--onefile", action="store_true", help="단일 EXE 로 묶음 (default 는 onedir).")
    parser.add_argument("--zip", action="store_true", help="onedir 빌드 후 dist 폴더를 ZIP 으로 자동 묶음.")
    parser.add_argument(
        "--with-splash",
        action="store_true",
        help="Enable the PyInstaller bootloader splash. Requires a working tkinter/Tcl-Tk build environment.",
    )
    parser.add_argument("--skip-smoke-test", action="store_true", help="빌드 후 자동 smoke test (exe import 검증) 건너뛰기.")
    return parser.parse_args()


def zip_onedir_bundle(root: Path) -> Path:
    """(2026-05-28) onedir 산출물 (`dist/{APP_NAME}/`) 을 ZIP 으로 묶음. USB / 메일 휴대용.
    Pillow 같은 추가 의존성 없이 stdlib shutil.make_archive 사용."""
    import datetime
    src_dir = root / "dist" / APP_NAME
    if not src_dir.is_dir():
        raise SystemExit(f"[ERROR] ZIP 생성: onedir 폴더가 없습니다 → {src_dir}")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = root / "dist" / f"{APP_NAME.replace(' ', '_')}_{stamp}"
    print(f"[RUN] Zipping {src_dir} → {out_base}.zip")
    zip_path_str = shutil.make_archive(
        base_name=str(out_base),
        format="zip",
        root_dir=str(src_dir.parent),
        base_dir=src_dir.name,
    )
    zip_path = Path(zip_path_str)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[OK] ZIP 생성 완료: {zip_path} ({size_mb:.1f} MB)")
    return zip_path


def smoke_test_built_exe(root: Path, onefile: bool) -> None:
    """(2026-05-28) 빌드된 exe 를 --smoke-test 로 실행 — 모든 import 가 frozen 환경에서
    실제로 통과하는지 검증. main.py 의 _run_smoke_test() 가 0/비0 으로 종료. 비0 이면
    빌드 실패로 처리해 잘못된 산출물이 배포 단계로 흘러가지 않도록 함."""
    if onefile:
        exe_path = root / "dist" / f"{APP_NAME}.exe"
    else:
        exe_path = root / "dist" / APP_NAME / f"{APP_NAME}.exe"
    if not exe_path.exists():
        raise SystemExit(f"[ERROR] Smoke test: exe not found at {exe_path}")
    print(f"[RUN] Smoke test: {exe_path} --smoke-test")
    try:
        result = subprocess.run(
            [str(exe_path), "--smoke-test"],
            cwd=str(exe_path.parent),
            timeout=120,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        raise SystemExit("[ERROR] Smoke test timed out after 120s.")
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr)
        raise SystemExit(f"[ERROR] Smoke test failed (exit code {result.returncode}). 빌드 산출물 사용 금지.")
    print("[OK] Smoke test passed — frozen EXE 의 핵심 import 가 모두 통과했습니다.")


def main() -> int:
    args = parse_args()
    root = project_root()
    # (2026-05-28) Default = onedir (첫 실행 빠르고, temp dir 경고/백신 false positive 회피).
    # --onefile 명시 시에만 단일 EXE. UPX off / 크래시 로그 / smoke test 같은 안전 옵션은 공통.
    if args.onefile and args.onedir:
        raise SystemExit("[ERROR] --onefile 과 --onedir 를 동시에 지정할 수 없습니다.")
    onefile = bool(args.onefile)

    print(f"[INFO] Project root: {root}")
    print(f"[INFO] Bundle mode: {'onefile (단일 EXE)' if onefile else 'onedir (default — 폴더 통째)'}")
    check_dependencies(skip_optional=args.skip_optional_check)
    splash_path = None
    if args.with_splash:
        if bootloader_splash_supported():
            splash_path = prepare_splash_image()
        else:
            raise SystemExit(
                "[ERROR] --with-splash was requested, but tkinter is not available in the build environment."
            )
    else:
        print("[INFO] Building without PyInstaller bootloader splash. Use --with-splash to enable it.")

    spec_path = write_spec(
        root,
        console=args.console,
        onefile=onefile,
        splash_image=splash_path,
    )
    if args.write_spec_only:
        print("[DONE] Spec generation completed.")
        return 0

    run_pyinstaller(root, spec_path, clean=args.clean)
    verify_no_host_runtime_contamination(root)
    normalize_onedir_qt_runtime(root, onefile=onefile)
    verify_build(root, onefile=onefile)
    if args.skip_smoke_test:
        print("[INFO] --skip-smoke-test 지정 — 빌드 후 검증 건너뜀.")
    else:
        smoke_test_built_exe(root, onefile=onefile)
    if args.zip:
        if onefile:
            print("[INFO] --zip 은 onedir 빌드에서만 의미 — onefile 모드에서는 무시.")
        else:
            zip_onedir_bundle(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
