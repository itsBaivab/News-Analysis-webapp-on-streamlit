FROM python:3.11.1

WORKDIR /app



COPY app.py /app/
COPY requirements.txt /app/
COPY . /app/
COPY count_vectorizer.pkl /app/
COPY naive_bayes_classifier.pkl /app/


RUN pip install -r requirements.txt

EXPOSE  8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT  ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address=0.0.0.0"]