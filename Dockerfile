FROM python:3.6-slim-stretch

WORKDIR /onnx_test

COPY requirements.txt /onnx_test

RUN pip install --upgrade pip &&  pip install https://cntk.ai/PythonWheel/CPU-Only/cntk-2.0-cp36-cp36m-linux_x86_64.whl &&pip install -r requirements.txt

COPY . /onnx_test