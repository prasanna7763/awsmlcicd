# # Use the official Python image from the 
# FROM python:3.10
# # Set working directory
# WORKDIR /app
# # Copy the requirements file into the image
# COPY requirements.txt .
# # Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the rest of the code into the image
# COPY . .

# # RUN jupyter nbconvert --to python dc_predictive_model.ipynb --output dc_predictive_model.py

# # Set the entrypoint to your training script
# ENTRYPOINT ["python", "test.py"]



# Use the official SageMaker Python SDK image
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.2-1

# Install necessary packages
RUN pip install --upgrade pip
RUN pip install pandas scikit-learn sagemaker boto3

# Set the working directory
WORKDIR /app

# Copy your training script into the container
COPY train_and_deploy.py .

# Set the entrypoint to run your script
ENTRYPOINT ["python", "train_and_deploy.py"]
