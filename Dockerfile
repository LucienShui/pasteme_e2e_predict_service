FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
CMD ["gunicorn", "app:app", "-c", "gunicorn.conf.py"]
