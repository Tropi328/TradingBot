FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home --shell /bin/bash botuser
USER botuser

ENTRYPOINT ["python", "main.py"]
CMD ["--dry-run"]
