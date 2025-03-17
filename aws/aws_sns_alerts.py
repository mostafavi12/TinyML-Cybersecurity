import boto3

sns_client = boto3.client("sns", region_name="us-east-1")

def send_alert(attack_message):
    response = sns_client.publish(
        TopicArn="arn:aws:sns:us-east-1:123456789012:CyberAttackAlerts",
        Message=attack_message,
        Subject="Cyberattack Detected!"
    )
    print("[✓] AWS SNS Alert Sent!")