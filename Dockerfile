FROM python:3.8
WORKDIR /app
COPY iris_model.pkl app.py /app/
RUN pip install flask flask-cors joblib numpy scikit-learn
CMD ["python", "app.py"]
