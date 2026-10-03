# src/engines/dsp_engine.py

import numpy as np
from scipy.signal import butter, filtfilt, welch

class DSPEngine:
    """
    동기화된 시계열 데이터에 수학적 연산 및 신호 처리를 적용하는 엔진입니다.
    """
    def __init__(self):
        pass

    def derivative(self, data: np.ndarray, dt: float) -> np.ndarray:
        """
        신호를 시간에 대해 미분합니다. (예: Position -> Velocity)
        np.gradient를 사용하여 데이터 양끝단의 오차를 최소화합니다.
        """
        if len(data) < 2:
            return data
        return np.gradient(data, dt)

    def low_pass_filter(self, data: np.ndarray, cutoff_hz: float, fs_hz: float, order: int = 4) -> np.ndarray:
        """
        버터워스(Butterworth) Low-pass 필터를 적용하여 고주파 노이즈를 제거합니다.
        filtfilt를 사용하여 위상 지연(Phase delay)이 0이 되도록 처리합니다. (제어 분석에 필수)
        
        :param data: 필터링할 원본 데이터 1차원 배열
        :param cutoff_hz: 잘라낼 기준 주파수 (예: 30Hz 이상 노이즈 제거)
        :param fs_hz: 데이터의 현재 샘플링 주파수 (예: 50Hz, 100Hz 등)
        """
        nyquist = 0.5 * fs_hz
        normal_cutoff = cutoff_hz / nyquist
        
        # 나이퀴스트 주파수를 초과하는 컷오프는 필터링 불가능하므로 원본 반환
        if normal_cutoff >= 1.0:
            print(f"경고: Cutoff({cutoff_hz}Hz)가 Nyquist({nyquist}Hz)보다 높아 필터를 적용하지 않습니다.")
            return data
            
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        # filtfilt: 데이터를 앞으로 한 번, 뒤로 한 번 적용하여 위상 밀림 현상을 상쇄
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    def calculate_psd(self, data: np.ndarray, fs_hz: float):
        """
        Welch의 방법을 사용하여 Power Spectral Density (전력 스펙트럼 밀도)를 계산합니다.
        특정 주파수(예: 기구 공진, 모터 회전수)에서 노이즈가 얼마나 강한지 파악할 때 씁니다.
        
        :return: (주파수 배열, PSD 값 배열)
        """
        # nperseg: FFT 윈도우 크기 (제어 주파수 분석 시 해상도를 위해 적절히 설정)
        nperseg = min(len(data), 1024) 
        freqs, psd = welch(data, fs=fs_hz, nperseg=nperseg)
        return freqs, psd

# === 단독 테스트 실행 (합성 데이터로 DSP 기능 검증) ===
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("[DSP Engine] 신호 처리 엔진 검증을 시작합니다...")
    
    dsp = DSPEngine()
    
    # 1. 가상의 비행 데이터 생성 (100Hz 샘플링, 1초 분량)
    fs = 100.0 
    dt = 1.0 / fs
    t = np.arange(0, 1.0, dt)
    
    # 진짜 기체의 움직임(2Hz의 부드러운 사인파) + 모터 진동 노이즈(30Hz의 떨림) 합성
    true_motion = np.sin(2 * np.pi * 2 * t)
    motor_noise = 0.5 * np.sin(2 * np.pi * 30 * t)
    raw_signal = true_motion + motor_noise
    
    # 2. Low-pass 필터 적용 (10Hz 이상의 노이즈를 깎아냄)
    cutoff = 10.0
    filtered_signal = dsp.low_pass_filter(raw_signal, cutoff_hz=cutoff, fs_hz=fs)
    
    # 3. 미분 적용 (필터링된 위치 데이터를 미분하여 속도 도출)
    velocity_signal = dsp.derivative(filtered_signal, dt)
    
    # 4. 결과 출력
    print("\n[수학 연산 완료]")
    print(f"원본 데이터 크기: {len(raw_signal)}")
    print(f"필터 적용 후 데이터 상위 5개: \n{filtered_signal[:5]}")
    print(f"미분 적용 후 데이터 상위 5개: \n{velocity_signal[:5]}")
    
    print("\n[안내] 실제 개발 환경에서는 이 엔진이 TimeSync 엔진과 결합되어")
    print("사용자가 'velocity = derivative(position)' 수식을 입력할 때 자동 호출됩니다.")