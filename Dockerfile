FROM vllm/vllm-openai@sha256:2e771fa615452282cc331eb418b3ef21636fce355bea0491fca89e6d362ab703

LABEL org.opencontainers.image.source="https://github.com/utensil/worker-vllm-glm"
LABEL org.opencontainers.image.description="Public vLLM image for GLM-5.3-Flash-NVFP4"

COPY patch_glm5next_processor.py /tmp/patch_glm5next_processor.py
RUN python3 /tmp/patch_glm5next_processor.py \
    && rm /tmp/patch_glm5next_processor.py
