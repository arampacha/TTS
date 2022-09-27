import argparse
import os
import requests

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, CharactersConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.speakers import SpeakerManager

parser = argparse.ArgumentParser()
parser.add_argument(
    '--task_id', required=True
)
parser.add_argument(
    '--model_name', required=False
)
parser.add_argument(
    '--finetuned_model', required=False
)

args = parser.parse_args()
task_id = args.task_id

# r = requests.post(
#     'https://tts-api-v0-iweeluwcja-uc.a.run.app//v1/trainingstatus', 
#     json={"uuid"=task_id, "status":"active"}
# )

BUCKET_NAME = os.environ.get('BUCKET_NAME')

# output_path = os.path.join('gcs', BUCKET_NAME, 'models', model_name)

dataset_config = BaseDatasetConfig(
    name="voicemode_vctk_plus", meta_file_train='train', path=f"/gcs/{BUCKET_NAME}/data", meta_file_val='dev',
)

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=None,
    eval_split_size=None,
)