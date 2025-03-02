import boto3


class S3:
    def __init__(self):
        self.s3 = boto3.client('s3')

    def upload_file(self, file_path: str, bucket: str, s3_key: str):
        self.s3.upload_file(file_path, bucket, s3_key)
