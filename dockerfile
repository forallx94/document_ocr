From pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

RUN apt-get update && apt-get install git -y 

RUN pip install uv 

ENV UV_SYSTEM_PYTHON=1

RUN uv pip install jupyter transformers==4.46.3 tokenizers==0.20.3 addict matplotlib einops easydict
RUN uv pip install flash-attn==2.7.3 --no-build-isolation