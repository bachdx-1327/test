ARG image_type=runtime
ARG cuda_version=12.2
ARG pytorch_version=2.2.2
FROM pytorch/pytorch:${pytorch_version}-cuda${cuda_version}-cudnn8-${image_type}

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt

WORKDIR /app
COPY app/* /app/

CMD ["python", "app.py"]

#RUN pip install -r requirements.txt