FROM nvcr.io/nvidia/pytorch:24.07-py3

RUN pip install --no-cache-dir \
    accelerate \
    transformers \
    evaluate==0.4.0 \
    datasets==2.3.2 \
    schedulefree \
    huggingface_hub>=0.20.0

COPY ./scripts /workspace

WORKDIR /workspace
