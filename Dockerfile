FROM python:3.7-slim
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
CMD python3 movie_classifier.py -t "$title" -d "$description"
