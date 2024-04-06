import logging
import os
from datetime import datetime

if not os.path.isdir('logs'):
    os.mkdir('logs')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=f'logs/training_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.log',
                    filemode='w')

logger = logging.getLogger(__name__)