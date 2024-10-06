import pandas as pd
import os
import sagemaker
import boto3
from sklearn.model_selection import train_test_split
from sagemaker.debugger import Rule, ProfilerRule, rule_configs
from sagemaker.session import TrainingInput
from sagemaker import image_uris
from sagemaker.serializers import CSVSerializer

def main():
    # Load your dataset here
    # Replace with your data loading logic
    # For example: df = pd.read_csv('your_data.csv')
    # X, y = df.drop(columns=['Income>50K']), df['Income>50K']
    
    # # Split the data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    
    # # Prepare train and validation datasets
    # train = pd.concat([pd.Series(y_train, index=X_train.index, name='Income>50K', dtype=int), X_train], axis=1)
    # validation = pd.concat([pd.Series(y_val, index=X_val.index, name='Income>50K', dtype=int), X_val], axis=1)

    # # Save datasets to CSV
    # train.to_csv('train.csv', index=False, header=False)
    # validation.to_csv('validation.csv', index=False, header=False)

    # # Set up S3 bucket
    # bucket = sagemaker.Session().default_bucket()
    # prefix = "demo-sagemaker-xgboost-adult-income-prediction"
    
    # # Upload data to S3
    # boto3.Session().resource('s3').Bucket(bucket).Object(
    #     os.path.join(prefix, 'data/train.csv')).upload_file('train.csv')
    # boto3.Session().resource('s3').Bucket(bucket).Object(
    #     os.path.join(prefix, 'data/validation.csv')).upload_file('validation.csv')

    # Get AWS Region and Role
    # region = sagemaker.Session().boto_region_name
    # role = sagemaker.get_execution_role()
    
    region = "us-west-2"
    role = "sagemakerRole"
    bucket = "sagemaker-us-west-2-533267114472"
    prefix = "demo-sagemaker-xgboost-adult-income-prediction"


    # Define S3 output location
    s3_output_location = f's3://{bucket}/{prefix}/xgboost_model'
   
    # Retrieve XGBoost container
    # container = image_uris.retrieve("xgboost", region, "1.2-1")

    container = "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:latest"


    # # Set up XGBoost model
    # xgb_model = sagemaker.estimator.Estimator(
    #     image_uri=container,
    #     role=role,
    #     instance_count=1,
    #     instance_type='ml.m4.xlarge',
    #     volume_size=5,
    #     output_path=s3_output_location,
    #     sagemaker_session=sagemaker.Session(),
    #     rules=[
    #         Rule.sagemaker(rule_configs.create_xgboost_report()),
    #         ProfilerRule.sagemaker(rule_configs.ProfilerReport())
    #     ]
    # )

    sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=region))

    xgb_model = sagemaker.estimator.Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type='ml.m4.xlarge',
    volume_size=5,
    output_path=s3_output_location,
    sagemaker_session=sagemaker_session,  # Pass the session with region
    rules=[
        Rule.sagemaker(rule_configs.create_xgboost_report()),
        ProfilerRule.sagemaker(rule_configs.ProfilerReport())
    ]
    )


    # Set hyperparameters
    xgb_model.set_hyperparameters(
        max_depth=5,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.7,
        objective="binary:logistic",
        num_round=1000
    )

    # Set up training input
    train_input = TrainingInput(
        f"s3://{bucket}/{prefix}/data/train.csv", content_type="csv"
    )
    validation_input = TrainingInput(
        f"s3://{bucket}/{prefix}/data/validation.csv", content_type="csv"
    )

    # Train the model
    xgb_model.fit({"train": train_input, "validation": validation_input}, wait=True)

    # Deploy the model
    xgb_predictor = xgb_model.deploy(
        initial_instance_count=1,
        instance_type='ml.t2.medium',
        serializer=CSVSerializer()
    )

    print("Model deployed successfully!")

if __name__ == "__main__":
    main()
