FROM pytorch/pytorch:2.13.0-cuda13.2-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends tmux asciinema && rm -rf /var/lib/apt/lists/*

ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN pip install --no-cache-dir {pip_requirements}
RUN python -c "import {import_name}"
ENV TVM_BACKTRACE=1

COPY docs/ /docs/

WORKDIR /app
COPY contract.py task.py test.sh /tests/
COPY clock.sh /etc/mkbench_clock.sh
RUN printf '. /etc/mkbench_clock.sh\n' >> /root/.bashrc
