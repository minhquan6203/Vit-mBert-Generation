import os
import yaml
import argparse
import logging
import json
from typing import Text
import torch
import transformers

from task.train import STVQA_Task


def main(config_path: Text) -> None:
    transformers.logging.set_verbosity_error()
    logging.basicConfig(level=logging.INFO)
    
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    
    logging.info("Training started...")
    task=STVQA_Task(config)
    task.training()
    logging.info("Training complete")
    
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    
    main(args.config)