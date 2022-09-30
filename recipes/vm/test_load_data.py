import argparse
import os
import requests

from smart_open import open

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, CharactersConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.speakers import SpeakerManager

import json

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     '--task_id', required=True
# )
# parser.add_argument(
#     '--model_name', required=False
# )
# parser.add_argument(
#     '--finetuned_model', required=False
# )

# args = parser.parse_args()
# task_id = args.task_id

task_id = 'test-0'

# r = requests.post(
#     'https://tts-api-v0-iweeluwcja-uc.a.run.app//v1/trainingstatus', 
#     json={"uuid"=task_id, "status":"active"}
# )

BUCKET_NAME = os.environ.get('BUCKET_NAME', 'vm-test-0')

# output_path = os.path.join('gcs', BUCKET_NAME, 'models', model_name)

dataset_config = BaseDatasetConfig(
    name="voicemod_userdata", meta_file_train='train', path=f"/gcs/{BUCKET_NAME}/data", meta_file_val='dev',
)

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True
)



with open(f"/gcs/{BUCKET_NAME}/models/test-0/eval.json", 'w') as f:
    json.dump(eval_samples)

try:
    sample = train_samples[0]
    audiofile = sample['audio_file']
    with open(audiofile) as f:
        f.read()
    read = "Done"
except Exception as e:
    read = e

with open(f"/gcs/{BUCKET_NAME}/models/test-0/debug.txt", 'w') as f:
    f.write(str(os.environ.get('AIP_MODEL_DIR'))+'\n')
    f.write(str(os.environ.get('AIP_MODEL_DIR'))+'\n')
    f.write("reading file: ", read + '\n')