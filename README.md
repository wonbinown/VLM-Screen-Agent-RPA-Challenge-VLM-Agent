# RPA Challenge VLM Agent

Streamlit과 OpenAI GPT-4o Vision API를 활용하여 RPA Challenge 화면의 입력 필드를 분석하고, OpenCV 후보 탐지와 GPT 후보 매칭을 통해 자동화 좌표를 시각화하는 프로토타입입니다.

이 프로젝트는 기존 RPA가 화면 변경, 입력 필드 위치 변경, 팝업 등 예외 상황에 취약하다는 문제에서 출발했습니다.  
특히 RPA Challenge처럼 입력 필드 위치가 매번 바뀌는 환경에서, VLM이 화면을 이해하고 자동화 액션을 도울 수 있는지 실험했습니다.

---

## 1. 프로젝트 목적

기존 RPA는 고정 좌표, XPath, CSS Selector 등에 의존하는 경우가 많습니다.  
하지만 화면 구조가 바뀌거나 입력 필드 위치가 바뀌면 자동화가 실패할 수 있습니다.

이 프로젝트의 목표는 다음과 같습니다.

- RPA Challenge 화면 이미지를 업로드한다.
- GPT-4o Vision이 화면을 이해하도록 한다.
- OpenCV로 입력선 후보를 탐지한다.
- GPT가 입력 필드와 후보를 매칭한다.
- Python 코드가 결과를 검증한다.
- Pillow로 최종 좌표를 이미지 위에 시각화한다.

---

## 2. 주요 기능

- RPA Challenge 화면 이미지 업로드
- OpenAI GPT-4o Vision API 연동
- OpenCV 기반 입력선 후보 탐지
- GPT-4o Vision 기반 필드-후보 매칭
- candidate_id 중복 검증
- 유효하지 않은 candidate_id 검증
- Pillow 기반 좌표 시각화
- Streamlit UI 제공

---

## 3. 기술 스택

| 구분 | 기술 |
|---|---|
| Language | Python |
| UI | Streamlit |
| Vision AI | OpenAI GPT-4o Vision |
| Image Processing | OpenCV, Pillow |
| Data Format | JSON |
| Automation Extension | PyAutoGUI 예정 |

---

## 4. 개발 과정

### 4.1 초기 접근: VLM 단독 좌표 추정

처음에는 GPT-4o Vision에게 RPA Challenge 화면 이미지를 전달하고, 7개 입력 필드의 좌표를 직접 JSON으로 반환하도록 구현했습니다.

대상 필드:

- First Name
- Last Name
- Company Name
- Role in Company
- Address
- Email
- Phone Number

하지만 실험 결과, VLM은 화면의 의미와 라벨은 어느 정도 이해했지만 입력창의 정확한 클릭 좌표를 안정적으로 반환하지 못했습니다.

발견한 문제:

- 입력선이 아니라 라벨 근처를 좌표로 반환
- Bounding Box가 과도하게 크게 잡힘
- Address와 Role in Company처럼 인접한 필드를 혼동
- 얇은 회색 밑줄 형태의 입력 필드를 정확히 탐지하지 못함

---

### 4.2 개선 접근: OpenCV 후보 탐지 + GPT 후보 매칭

VLM이 직접 좌표를 추정하는 방식의 한계를 보완하기 위해 구조를 변경했습니다.

개선된 구조:

```text
RPA Challenge 화면 이미지
↓
OpenCV로 입력선 후보 탐지
↓
후보마다 candidate_id 부여
↓
GPT-4o Vision이 field_name과 candidate_id를 매칭
↓
Python이 중복/유효성 검증
↓
Pillow로 최종 좌표 시각화