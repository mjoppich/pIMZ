FROM python:3.8-slim

WORKDIR /app/
COPY . .

RUN apt-get update \
&& apt-get install -y build-essential gcc zlib1g-dev \
&& apt-get clean

RUN pip install -e .
RUN pip install jupyter
CMD jupyter-notebook --ip="*" --no-browser --allow-root
