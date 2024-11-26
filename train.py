from engine.model import Model
from loguru import logger

def main():
    model = Model(
        model='./cfg/base.yaml',
        task='pose',
        verbose=False
    )
