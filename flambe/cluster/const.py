# Logging socket config
SOCKET_TIMEOUT = 50

# SSH connection settings
RETRY_DELAY = 1
RETRIES = 60

TENSORBOARD_IMAGE = "tensorflow/tensorflow:1.15.0"

RAY_REDIS_PORT = 12345

# This names are specific because of tune stuff!
# NOTE: DO NOT CHANGE THIS FILE NAMES
# TODO: do this in a cleaner way

PRIVATE_KEY = "ray_bootstrap_key.pem"
PUBLIC_KEY = "ray_bootstrap_key.pub"

REPORT_SITE_PORT = 49558
TENSORBOARD_PORT = 49556

# This is the account number for Flambe AWS account.
# There is not risk of it being public.
AWS_FLAMBE_ACCOUNT = "808129580301"
