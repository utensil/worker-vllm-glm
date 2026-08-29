FROM vllm/vllm-openai@sha256:2e771fa615452282cc331eb418b3ef21636fce355bea0491fca89e6d362ab703 AS base

LABEL org.opencontainers.image.source="https://github.com/utensil/worker-vllm-glm"
LABEL org.opencontainers.image.description="Public vLLM image for GLM-5.3-Flash-NVFP4"

COPY patch_glm5next_processor.py /tmp/patch_glm5next_processor.py
RUN python3 /tmp/patch_glm5next_processor.py \
    && rm /tmp/patch_glm5next_processor.py

# Queue-worker image. It owns a loopback-only vLLM server and does not register
# with RunPod until that server is healthy. Build explicitly with
# `--target serverless`; the final stage remains the existing Pod image.
FROM base AS serverless

COPY serverless/requirements.txt /tmp/serverless-requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/serverless-requirements.txt \
    && rm /tmp/serverless-requirements.txt

COPY serverless /opt/glm-serverless
ENTRYPOINT ["python3", "-u", "/opt/glm-serverless/main.py"]

# Keep this as the final/default target so existing Pod builds preserve their
# entrypoint and behavior. The public Pod template remains digest-pinned.
FROM base AS pod
