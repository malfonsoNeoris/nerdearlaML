import os
import yaml
import logging
import logging.config


def configure_logging(logging_file="logging_configs/logging_debug.yaml", env_key="LOG_CFG"):
	"""
	Setup logging configuration
	"""
	path = logging_file
	#value = os.getenv(env_key, None)
	value = None
	if value:
		path = value
	if path is not None and os.path.exists(path):
		with open(path, "rt") as in_file:
			config = yaml.safe_load(in_file.read())
		logging.config.dictConfig(config)
	else:
		logging.basicConfig(level=logging.INFO)