FROM tiangolo/uwsgi-nginx-flask:python3.8

RUN pip install numpy pandas joblib scikit-learn 

COPY ./app /app

WORKDIR /app

RUN python3 train.py


