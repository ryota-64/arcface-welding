FROM pytorch/pytorch
RUN apt-get update && apt-get install -y libopencv-dev
COPY requirements.txt /workspace/
RUN pip install -r requirements.txt
