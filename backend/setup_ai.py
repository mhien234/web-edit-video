"""Tải gói dịch offline cho Argos Translate.

Ví dụ:  python -m backend.setup_ai en vi      (dịch Anh -> Việt)
        python -m backend.setup_ai            (mặc định: en<->vi, ja->en, ko->en)
Cần có mạng ở lần chạy này; sau đó dịch hoàn toàn offline.
"""
from __future__ import annotations

import sys


def main(pairs: list[tuple[str, str]]) -> None:
    try:
        from argostranslate import package
    except ImportError:
        sys.exit("Chưa cài argostranslate. Chạy: pip install -r requirements-ai.txt")
    print("Đang tải danh sách gói dịch...")
    package.update_package_index()
    available = package.get_available_packages()
    for src, dst in pairs:
        pkg = next((p for p in available if p.from_code == src and p.to_code == dst), None)
        if not pkg:
            print(f"  - Không có gói {src}->{dst} (có thể dịch bắc cầu qua tiếng Anh nếu có {src}->en và en->{dst})")
            continue
        print(f"  - Đang cài {src}->{dst} ...")
        package.install_from_path(pkg.download())
    print("Xong.")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 2:
        main([(args[0], args[1])])
    else:
        main([("en", "vi"), ("vi", "en"), ("ja", "en"), ("ko", "en")])
