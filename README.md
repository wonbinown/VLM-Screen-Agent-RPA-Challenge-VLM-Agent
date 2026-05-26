# RPA Challenge VLM Agent

Streamlit과 OpenAI GPT-4o Vision API를 활용하여 RPA Challenge 화면의 입력 필드를 분석하고, OpenCV 후보 탐지와 GPT 후보 매칭을 통해 자동화 좌표를 시각화하는 프로토타입입니다.

## 주요 기능

- RPA Challenge 화면 이미지 업로드
- OpenCV 기반 입력선 후보 탐지
- GPT-4o Vision 기반 필드-후보 매칭
- candidate_id 중복 검증
- Pillow 기반 좌표 시각화
- Streamlit UI 제공

## 현재 상태

- VLM 단독 좌표 추정의 한계 확인
- OpenCV 후보 탐지 + GPT 매칭 구조로 개선
- 시스템 검증 레이어 추가
- 후보 품질 개선은 향후 과제로 남김