목표는 document ocr 을 이용하여 데이터 파싱을 얼만큼 할 수 있는지 확인 및 성능 확인 하는 작업을 진행 하는 것

손글씨 및 프린트된 글자가 혼합되어 있는 문서에서 명확한 타겟을 정확하게 찾을 수 있는가? 에 대한 질문을 확인하기 위한 목적

1. document ocr, 진행 후 sllm 으로 후처리
2. document ocr, 진행 후 regex 으로 후처리
3. object detection 으로 목표 탐색 후 , ocr

3 가지 방식이 고려되나 2,3 은 비교적 쉽게 될 수 있으며 성능에 대한 감각적인 판단이 되나
1번에 대한 확인이 들지 않아 실험으로 확인하고자 함

사용 데이터는 ai hub 의 [공공행정문서 OCR](https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=OCr&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&srchDataRealmCode=REALM002&aihubDataSe=data&dataSetSn=88) 을 활용 전체 데이터가 필요하지 않음으로 샘플 데이터를 활용하여 테스트

모델은 1. DeepSeek OCR-2 , 2. PaddleOCR VL, 3. gemini-3 flash 
대다수의 실제 프로젝트는 데이터의 보안 관련 문제로 api 형 모델을 사용할 수 없으나
실제 프로젝트 시작전 고객들을 gpt, gemini, grok 같은 모델로 테스트를 진행 할 것으로 보여 비교 군으로 설정