# Use the official Python image from the 
FROM python:3.10
# Set working directory
WORKDIR /app
# Copy the requirements file into the image
COPY requirements.txt .
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code into the image
COPY . .

# RUN jupyter nbconvert --to python dc_predictive_model.ipynb --output dc_predictive_model.py

# Set the entrypoint to your training script
ENTRYPOINT ["python", "test.py"]