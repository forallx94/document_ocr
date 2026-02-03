import torch
import time
import os
import math
from transformers import AutoModel, AutoTokenizer
from multiprocessing import Process, set_start_method

# 전역 설정
MODEL_NAME = 'deepseek-ai/DeepSeek-OCR-2'
INPUT_FOLDER = 'data/New_sample/원천데이터/인.허가/5350109/1994/'
OUTPUT_BASE_PATH = '/workspace/output/deepseek-ocr-2/1994/'
GPU_IDS = [0, 1, 2]  # 사용할 GPU 번호들

def worker_process(gpu_id, file_list):
    """
    각 GPU에서 실행될 개별 프로세스 함수
    """
    # 해당 프로세스가 사용할 GPU 지정
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"[GPU {gpu_id}] 모델 로딩 시작 (할당량: {len(file_list)}장)")
    
    try:
        # 모델 및 토크나이저 로드 (각 프로세스 독립적)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            MODEL_NAME, 
            attn_implementation="eager", 
            trust_remote_code=True, 
            use_safetensors=True
        )
        model = model.eval().cuda().to(torch.bfloat16)
        
        for filename in file_list:
            image_path = os.path.join(INPUT_FOLDER, filename)
            save_name = os.path.splitext(filename)[0]
            output_path = os.path.join(OUTPUT_BASE_PATH, save_name)
            os.makedirs(output_path, exist_ok=True)

            prompt = "<image>\n<|grounding|>Convert the document to markdown. "
            
            try:
                torch.cuda.reset_peak_memory_stats()
                start_time = time.time()
                
                # 추론
                model.infer(
                    tokenizer, 
                    prompt=prompt, 
                    image_file=image_path, 
                    output_path=output_path, 
                    base_size=1024, 
                    image_size=768, 
                    crop_mode=True, 
                    save_results=True
                )
                
                duration = time.time() - start_time
                max_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)

                # 개별 리포트 저장
                with open(os.path.join(output_path, "individual_performance.txt"), "w") as f:
                    f.write(f"GPU ID: {gpu_id}\n")
                    f.write(f"File: {filename}\n")
                    f.write(f"Time: {duration:.22f}s\n")
                    f.write(f"Peak VRAM: {max_vram:.2f}GB\n")
                
                print(f"[GPU {gpu_id}] 성공: {filename} ({duration:.1f}s)")

            except Exception as e:
                print(f"[GPU {gpu_id} 오류 - {filename}] {e}")
                continue
                
            # 주기적인 캐시 정리
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"[GPU {gpu_id} 치명적 오류] 모델 로드 실패: {e}")

def main():
    # 처리할 파일 목록 확보
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    all_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(valid_extensions)])
    
    if not all_files:
        print("처리할 이미지가 없습니다.")
        return

    print(f"총 {len(all_files)}개의 이미지를 {len(GPU_IDS)}대의 GPU로 분할 처리합니다.")

    # 파일을 GPU 개수만큼 균등하게 분할
    num_gpus = len(GPU_IDS)
    chunk_size = math.ceil(len(all_files) / num_gpus)
    chunks = [all_files[i:i + chunk_size] for i in range(0, len(all_files), chunk_size)]

    processes = []
    
    # 각 GPU별로 프로세스 생성 및 시작
    for i, gpu_id in enumerate(GPU_IDS):
        if i < len(chunks):
            p = Process(target=worker_process, args=(gpu_id, chunks[i]))
            p.start()
            processes.append(p)
            time.sleep(5) # 모델 로딩 시 VRAM 피크 충돌 방지를 위한 간격

    # 모든 프로세스 종료 대기
    for p in processes:
        p.join()

    print("\n[모든 작업이 완료되었습니다.]")

if __name__ == '__main__':
    # CUDA 멀티프로세싱 안정성을 위해 'spawn' 방식 사용
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()