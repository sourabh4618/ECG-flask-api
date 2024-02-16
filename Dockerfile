# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:latest

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Adjusted to the Flask apps port
EXPOSE 5000  

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN pip install -r requirements.txt
RUN python -m pip install --upgrade pip

# Run the Flask app on container startup.
CMD ["python", "app.py"]
