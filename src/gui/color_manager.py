# src/gui/color_manager.py

class ColorManager:
    """
    모든 그래프의 색상 규칙을 통제합니다.
    - Setpoint (목표값): 빨간색 (Red)
    - Actual (실제/측정값): 파란색 (Blue)
    """
    @staticmethod
    def get_color(topic_name, signal_name):
        t = topic_name.lower()
        s = signal_name.lower()
        
        # 1. Setpoint 판별 (토픽명이나 변수명에 힌트가 있는 경우)
        if "setpoint" in t or "_sp" in s or "q_d" in s:
            return "#FF3333"  # Red
            
        # 2. Actual 판별 (그 외 모든 측정값)
        return "#3388FF"  # Blue