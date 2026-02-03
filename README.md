# Document OCR Performance Analysis: sLLM vs. Traditional Methods

본 프로젝트는 **손글씨와 인쇄체(Printed)가 혼합된 공공행정문서**에서 특정 타겟 데이터를 정확하게 추출하기 위한 OCR 및 후처리 방법론의 성능을 검증합니다. 특히, 최근 주목받는 VLM(Vision Language Model) 및 sLLM 기반의 후처리 방식이 실무 환경에서 어느 정도의 효용성을 가지는지 확인하는 데 목적이 있습니다.

## 1. 실험 목적
- **핵심 질문**: "복합 문서(손글씨+인쇄체)에서 특정 필드 값을 정확하게 파싱할 수 있는가?"
- **검증 대상**:
    1. **Document OCR + sLLM 후처리** (본 실험의 주안점)
    2. Document OCR + Regex(정규표현식) 후처리
    3. Object Detection(타겟 영역 검출) + OCR
- **보안 요구사항**: 실제 프로젝트 환경의 데이터 보안을 고려하여 온프레미스(On-premise) 구동 모델(DeepSeek, PaddleOCR)을 주 모델로 사용하며, 성능 비교군으로 상용 API(Gemini)를 활용합니다.

## 2. 데이터셋 및 환경
- **Dataset**: [AI Hub 공공행정문서 OCR](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=88) 샘플 데이터
- **Models**:
  - `DeepSeek-OCR-2`
  - `PaddleOCR-VL 1.5`
  - `Gemini 3 Flash` (비교 벤치마크용)

---

## 3. 실험 결과 요약 (Benchmark Summary)

| Metric | DeepSeek-OCR-2 | PaddleOCR-VL | 비고 |
| :--- | :---: | :---: | :--- |
| **대상 이미지 수** | 100 장 | 89 장 | PaddleOCR은 시간 이슈로 중도 중단 |
| **평균 처리 시간 (sec)** | **53.66 s** | 1246.94 s | DeepSeek이 압도적으로 빠름 |
| **VRAM 평균 사용량** | 10.78 GB | **4.07 GB** | 메모리 최적화 실패로 예상 |
| **평균 CER (낮을수록 우수)** | 1.4016 | **1.1244** | 문자 인식률은 PaddleOCR이 근소 우위 |
| **평균 Accuracy (%)** | **45.84 %** | 28.38 % | 전체 문장 완성도는 DeepSeek이 우수 |

---

## 4. 상세 로그 (Performance Logs)

### [Case 1] DeepSeek-OCR-2
DeepSeek은 전반적으로 일관된 VRAM 점유율과 상대적으로 빠른 처리 속도를 보여주었습니다. 특히 Accuracy 측면에서 PaddleOCR보다 높은 성적을 거두어, sLLM 후처리에 적합한 문맥 파악 능력을 시사합니다.

```text
==================================================
      DeepSeek-OCR-2 Overall Performance Log      
==================================================
- Mean Time    : 53.6665 sec
- VRAM Usage   : 10.78 GB (Static)
- Avg CER      : 1.4016
- Avg Accuracy : 45.84 %
- Max CER      : 23.4934 / Min CER : 0.2133
```

### [Case 2] PaddleOCR-VL
PaddleOCR-VL은 VRAM 사용량은 적으나, 추론 속도에서 심각한 병목 현상이 발생했습니다. 병렬 처리를 시도했음에도 불구하고 이미지 한 장당 평균 20분 이상 소요되어 실시간성이 필요한 프로젝트에는 부적합한 것으로 판단됩니다.

```text
==================================================
       PaddleOCR-VL Overall Performance Log       
==================================================
- Mean Time    : 1246.94 sec (Performance Issue)
- VRAM Usage   : 4.07 GB
- Avg CER      : 1.1244
- Avg Accuracy : 28.38 %
- Max CER      : 3.9537 / Min CER : 0.1891
```

---

## 5. 실험적 인사이트 (중요 지표)

1.  **처리 효율성**: DeepSeek-OCR-2는 PaddleOCR-VL 대비 약 23배 빠른 처리 속도를 보여주었습니다. PaddleOCR의 경우 병렬 처리를 적용했음에도 불구하고 속도 저하가 심해 코드 최적화 혹은 모델 경량화가 선행되어야 합니다.
2.  **문맥 파악 및 구조화**: CER(문자 오류율)은 두 모델이 유사(1.1~1.4)했으나, Accuracy(정확도)는 DeepSeek이 월등히 높았습니다(45.8% vs 28.3%). 이는 DeepSeek이 문서의 구조를 유지하며 데이터를 파싱하는 능력이 상대적으로 우수함을 나타냅니다.
3.  **손글씨 인식의 한계 (Critical Issue)**:
    *   **인식 불가**: 실험 결과, **두 모델 모두 손글씨 영역에 대해서는 정상적인 인식을 수행하지 못하는 것**으로 확인되었습니다. 
    *   **프로젝트 타당성**: 행정 문서의 특성상 손글씨 데이터의 비중이 높으나, 현재의 Zero-shot 성능으로는 실제 상용 프로젝트에 적용하기에 매우 어려운 수준입니다.
4.  **재학습(Fine-tuning)의 불확실성**:
    *   손글씨 데이터셋을 활용한 추가 학습(Fine-tuning)을 고려할 수 있으나, 현재의 베이스 모델 성능을 고려할 때 재학습 후의 성능 향상 폭이 실무 요구치를 충족할 수 있을지에 대해서는 의문이 제기됩니다. 

---

## 6. 향후 계획
- **Gemini 3 Flash 결과 비교**: 상용 모델과의 성능 격차를 확인하여 온프레미스 모델의 최적화 목표 설정.
- **후처리 로직 적용**: 추출된 Raw Text를 sLLM에 프롬프팅하여 JSON 형태로 정형화하는 테스트 진행.
- **Object Detection 연계**: 문서 내 특정 영역(도장, 서명, 특정 서식 함)을 먼저 찾고 OCR을 수행하는 3번 방식과의 하이브리드 검토.
- **데이터 증강 및 학습 전략**: 재학습 효율성을 검증하기 위한 소규모 PoC 학습 진행 여부 결정.

