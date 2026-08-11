# Execution sandbox for candidate-generated code.
#
# Deliberately minimal: CPU-only torch, pytest, nothing else. No compilers, no
# network clients, no shell utilities beyond what the base image ships. The
# smaller the image, the smaller the escape surface and the less there is to
# audit.
FROM python:3.12-slim

# Non-root from the start. Candidate code never runs as uid 0, so even a
# container escape lands on an unprivileged account.
RUN groupadd --gid 10001 runner \
 && useradd --uid 10001 --gid runner --create-home --shell /usr/sbin/nologin runner

# CPU wheels only. The GPU wheel is ~2 GB and this project does not train on GPU.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      "torch==2.9.1" \
 && pip install --no-cache-dir "pytest==8.3.4" "numpy==2.2.1"

# Where the grader mounts things. Both are bind-mounted at run time:
#   /grader    read-only : the check suite and the protected gold oracle
#   /workspace read-write: the candidate's repo, and nothing else
RUN mkdir -p /grader /workspace && chown runner:runner /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/grader \
    HOME=/home/runner

USER runner
WORKDIR /workspace

# No default command: the runner always passes an explicit one, so an
# accidentally-started container does nothing.
CMD ["python", "-c", "print('sandbox image ok')"]
