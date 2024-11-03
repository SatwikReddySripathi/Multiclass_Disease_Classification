# logging_config.py
import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
from typing import Dict, Any
from airflow.hooks.base import BaseHook
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

class PipelineLogger:
    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.log_dir = f"logs/{pipeline_name}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up file handler
        log_file = f"{self.log_dir}/{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        
        # Set up console handler
        console_handler = logging.StreamHandler()
        
        # Create formatters and add it to handlers
        log_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)
        
        # Create logger
        self.logger = logging.getLogger(pipeline_name)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Slack client setup
        self.slack_client = self._setup_slack()
        
    def _setup_slack(self) -> WebClient:
        """Set up Slack client using Airflow connection"""
        try:
            slack_conn = BaseHook.get_connection('slack_webhook')
            return WebClient(token=slack_conn.password)
        except Exception as e:
            self.logger.error(f"Failed to setup Slack client: {str(e)}")
            return None
    
    def log_task_start(self, task_name: str, params: Dict[str, Any] = None):
        """Log task start with parameters"""
        msg = f"Starting task: {task_name}"
        if params:
            msg += f" with parameters: {params}"
        self.logger.info(msg)
        
    def log_task_completion(self, task_name: str, metrics: Dict[str, Any] = None):
        """Log task completion with metrics"""
        msg = f"Completed task: {task_name}"
        if metrics:
            msg += f" with metrics: {metrics}"
        self.logger.info(msg)
        
    def log_error(self, task_name: str, error: Exception, alert: bool = True):
        """Log error and optionally send alert"""
        error_msg = f"Error in task {task_name}: {str(error)}"
        self.logger.error(error_msg)
        
        if alert and self.slack_client:
            try:
                self.slack_client.chat_postMessage(
                    channel="#pipeline-alerts",
                    text=f":red_circle: *Pipeline Error*\n{error_msg}"
                )
            except SlackApiError as e:
                self.logger.error(f"Failed to send Slack alert: {str(e)}")
                
    def log_metric(self, metric_name: str, value: Any):
        """Log a specific metric"""
        self.logger.info(f"Metric - {metric_name}: {value}")
