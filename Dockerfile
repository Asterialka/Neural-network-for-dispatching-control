FROM python:3.10-slim

# Install tkinter and other necessary dependencies
RUN apt-get update && apt-get install -y python3-tk && apt-get clean

WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application files to the working directory
COPY . .

CMD ["python", "app.py"]
