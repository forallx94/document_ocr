import json
import os
import re
import Levenshtein
import argparse

def preprocess_markdown(md_text):
    """
    Markdown 및 HTML 태그를 제거하고 순수 텍스트만 추출하여 정규화
    """
    # 1. HTML 태그 제거 (<table>, <tr>, <td> 등)
    clean_text = re.sub(r'<[^>]+>', ' ', md_text)
    # 2. 마크다운 특수 기호 제거 (|, -, #, *)
    clean_text = re.sub(r'[\|\-#\*]', ' ', clean_text)
    # 3. 연속된 공백 및 줄바꿈을 하나의 공백으로 통합
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

def extract_gt_text_from_json(json_path):
    """
    JSON 정답지에서 bbox 좌표 순서대로 텍스트 추출
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        annotations = data.get('annotations', [])
        if not annotations:
            return ""
        
        # 좌표(y 우선, x 차선) 순서로 정렬하여 시각적 읽기 순서 재구성
        # bbox 형식: [x, y, width, height]
        annotations.sort(key=lambda a: (a['annotation.bbox'][1], a['annotation.bbox'][0]))
        
        gt_text = " ".join([ann['annotation.text'] for ann in annotations])
        gt_text = re.sub(r'\s+', ' ', gt_text).strip()
        return gt_text
    except Exception as e:
        print(f"Error parsing JSON {json_path}: {e}")
        return ""

def calculate_cer_metrics(gt_text, pred_text):
    """
    CER (Character Error Rate) 및 Accuracy 계산
    """
    if not gt_text:
        return 1.0, 0.0 if pred_text else 0.0, 100.0
    
    edit_dist = Levenshtein.distance(gt_text, pred_text)
    cer = edit_dist / len(gt_text)
    accuracy = max(0, (1 - cer) * 100)
    return edit_dist, cer, accuracy

def main():
    # 경로 설정 (사용자 환경에 맞게 고정)
    GT_BASE_DIR = 'data/New_sample/라벨링데이터/인.허가/5350109/1994/'
    PRED_BASE_DIR = '/workspace/output/deepseek-ocr-2/1994/'
    REPORT_PATH = os.path.join(PRED_BASE_DIR, "cer_performance_report.txt")

    results = []
    
    print(f"--- CER 계산 시작 ---")
    print(f"정답지 경로: {GT_BASE_DIR}")
    print(f"예측결과 경로: {PRED_BASE_DIR}")

    # 결과 폴더 목록 가져오기
    if not os.path.exists(PRED_BASE_DIR):
        print(f"오류: 예측 결과 경로가 존재하지 않습니다: {PRED_BASE_DIR}")
        return

    pred_folders = sorted([f for f in os.listdir(PRED_BASE_DIR) if os.path.isdir(os.path.join(PRED_BASE_DIR, f))])

    for folder_name in pred_folders:
        pred_folder_path = os.path.join(PRED_BASE_DIR, folder_name)
        mmd_file_path = os.path.join(pred_folder_path, 'result.mmd')
        json_file_path = os.path.join(GT_BASE_DIR, f"{folder_name}.json")

        if os.path.exists(mmd_file_path) and os.path.exists(json_file_path):
            # 1. 정답 텍스트 추출
            gt_clean = extract_gt_text_from_json(json_file_path)
            
            # 2. 결과 텍스트 추출 및 전처리
            try:
                with open(mmd_file_path, 'r', encoding='utf-8') as f:
                    pred_raw = f.read()
                pred_clean = preprocess_markdown(pred_raw)
                
                # 3. CER 계산
                edit_dist, cer, acc = calculate_cer_metrics(gt_clean, pred_clean)
                
                results.append({
                    "filename": folder_name,
                    "gt_len": len(gt_clean),
                    "pred_len": len(pred_clean),
                    "edit_dist": edit_dist,
                    "cer": cer,
                    "accuracy": acc
                })
                print(f"Processed: {folder_name} | Acc: {acc:.2f}%")
                
            except Exception as e:
                print(f"Error processing {folder_name}: {e}")
        else:
            # 매칭되는 파일이 없는 경우 출력
            if not os.path.exists(json_file_path):
                # 가끔 이미지 파일명에 대응하는 JSON 파일명이 다를 수 있으므로 체크용
                pass 

    # 4. 최종 리포트 작성
    if results:
        total_gt_len = sum(r['gt_len'] for r in results)
        total_edit_dist = sum(r['edit_dist'] for r in results)
        
        # 전체 CER은 단순히 평균을 내는 것보다 전체 편집거리/전체 글자수로 계산하는 것이 정확함
        overall_cer = total_edit_dist / total_gt_len if total_gt_len > 0 else 0
        overall_acc = (1 - overall_cer) * 100
        
        with open(REPORT_PATH, "w", encoding='utf-8') as f:
            f.write("==============================================\n")
            f.write("       DeepSeek-OCR-2 CER Performance Report   \n")
            f.write("==============================================\n")
            f.write(f"Total Files Processed : {len(results)}\n")
            f.write(f"Overall Average Accuracy: {overall_acc:.2f}%\n")
            f.write(f"Overall Average CER     : {overall_cer:.4f}\n")
            f.write("----------------------------------------------\n")
            f.write(f"{'Filename':<40} | {'CER':<8} | {'Accuracy':<10}\n")
            f.write("-" * 65 + "\n")
            for r in results:
                f.write(f"{r['filename']:<40} | {r['cer']:<8.4f} | {r['accuracy']:<10.2f}%\n")
        
        print(f"\n[검증 완료] 리포트 저장됨: {REPORT_PATH}")
        print(f"최종 정확도: {overall_acc:.2f}%")
    else:
        print("\n[알림] 매칭된 파일이 없어 리포트를 생성하지 못했습니다.")

if __name__ == "__main__":
    main()