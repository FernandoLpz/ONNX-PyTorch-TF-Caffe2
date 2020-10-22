FROM python:3.6-slim-stretch

WORKDIR /onnx_test

COPY requirements.txt /onnx_test

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /onnx_test