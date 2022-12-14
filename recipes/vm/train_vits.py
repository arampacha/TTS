import argparse
import json
import os
from pathlib import Path
import requests


import torch
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
parser.add_argument('--name', type=str, default='testing')
parser.add_argument(
    '--task_id', type=str, required=True
)
parser.add_argument(
    '--restore_path', type=str, required=False, default=''
)
parser.add_argument(
    '--parent_model_dir', type=str, required=False, default=''
)
parser.add_argument(
    '--tgt_speaker', type=str, required=True
)
parser.add_argument(
    '--epochs', type=int, default=1
)
parser.add_argument(
    '--batch_size', type=int, default=2
)
parser.add_argument(
    '--eval_batch_size', type=int, default=2
)
parser.add_argument(
    '--batch_group_size', type=int, default=5
)
parser.add_argument(
    '--num_loader_workers', type=int, default=4,
)
parser.add_argument(
    '--save_step', type=int, default=1000
)
parser.add_argument(
    '--print_step', type=int, default=1000
)
parser.add_argument(
    '--logger', type=str, choices=['tensorboard'], default='tensorboard'
)
parser.add_argument(
    '--phoneme_cache_path', type=str, default=None
)

args = parser.parse_args()
task_id = args.task_id
speaker = args.tgt_speaker


def convert_gcs_path(path:str):
    if path.startswith("gs://"):
        return path.replace('gs:/', '/gcs', 1)

def log_task_status_upd(status_code:int):
    if status_code == 200:
        print("Updated task status")
    else:
        print("Failed to update task status")

model_dir = convert_gcs_path(os.environ.get('AIP_MODEL_DIR'))
checkpoint_dir = convert_gcs_path(os.environ.get('AIP_CHECKPOINT_DIR'))
tensorboard_dir = convert_gcs_path(os.environ.get('AIP_TENSORBOARD_LOG_DIR'))
parent_api_url = os.environ.get("PARENT_API_URL")
x_api_key = os.environ.get('X_API_KEY')
BUCKET_NAME = os.environ.get('BUCKET_NAME')
if BUCKET_NAME is None:
    raise ValueError("No `BUCKET_NAME` provided.")

# if checkpoint exists for this task continue from it
if os.path.isdir(checkpoint_dir):
    runs = sorted(
        [path.as_posix() for path in Path(checkpoint_dir).iterdir() if (path.is_dir() and path.name != 'phoneme_cache')],
        key=os.path.getmtime,
    )
    print("Resuming from checkpoint")
    if len(runs):
        continue_path = runs[-1]
    else:
        continue_path = None
else:
    continue_path = None

r = requests.post(
    f'{parent_api_url}/v1/trainingstatus', 
    json={"uuid":task_id, "status":"active"},
    headers={'x-api-key':x_api_key}
)
log_task_status_upd(r.status_code)


def create_speakers_file(model_dir:str, added_speaker:list, parent_model_dir:str=None):
    
    if parent_model_dir:
        with open(os.path.join(parent_model_dir, 'speakers.json')) as f:
            speakers = json.load(f)
    else:
        speakers = {}
    for speaker in added_speaker:
        speakers[speaker] = len(speakers)
    
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'speakers.json'), 'w') as f:
        json.dump(speakers, f)
    print(f"Saved updated speakers.json to {model_dir}")

def repackage_model(input_path, output_path):
    state = torch.load(input_path, map_location='cpu')

    state.keys()

    del state['optimizer']
    del state['scaler']

    for k in list(state['model'].keys()):
        if k.startswith('disc.'):
            del state['model'][k]

    torch.save(state, output_path)
    print(f"Saved model to {output_path}")

def create_infrence_config(input_path, output_path):

    with open(input_path) as f:
        config = json.load(f)
    
    config['model_args']['init_discriminator'] = False
    config['speakers_file'] = "./speakers.json"
    with open(output_path, 'w') as f:
        json.dump(config, f)
    print(f"Saved config to {output_path}") 

pretraining_dataset_config = BaseDatasetConfig(
    formatter="vctk_freeman", meta_file_train=f'train', path=f"/gcs/{BUCKET_NAME}/data", meta_file_val=f'dev', ignored_speakers=['s5', 'p315']
)
user_dataset_config = BaseDatasetConfig(
    formatter="voicemod_userdata", meta_file_train=f'train_{task_id}', path=f"/gcs/{BUCKET_NAME}/data", meta_file_val=f'dev_{task_id}',
)
audio_config = BaseAudioConfig(
    sample_rate=22050,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    preemphasis=0.0,
    ref_level_db=20,
    log_func="np.log",
    do_trim_silence=True,
    trim_db=45,
    mel_fmin=0,
    mel_fmax=None,
    spec_gain=1.0,
    signal_norm=False,
    do_amp_to_db_linear=False,
)

model_args = VitsArgs(
    use_speaker_embedding=True,
)

if args.phoneme_cache_path is not None:
    phoneme_cache_path = os.path.join(args.phoneme_cache_path)
    os.makedirs(phoneme_cache_path, exist_ok=True)
else:
    phoneme_cache_path=os.path.join(checkpoint_dir, "phoneme_cache")
config = VitsConfig(
    output_path = checkpoint_dir,
    model_args=model_args,
    audio=audio_config,
    run_name=args.name,
    run_description=f"Fine-tune VITS run {task_id}",
    project_name="VM",
    wandb_entity=None,
    batch_size=args.batch_size,
    eval_batch_size=args.eval_batch_size,
    batch_group_size=args.batch_group_size,
    num_loader_workers=args.num_loader_workers,
    num_eval_loader_workers=args.num_loader_workers,
    run_eval=True,
    test_delay_epochs=0,
    epochs=args.epochs,
    save_step=args.save_step,
    text_cleaner="english_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phonemizer="espeak",
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="_",
        eos="",
        bos="",
        characters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        punctuations=";:,.!?¡¿—…\"«»“” ",
        phonemes="ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
    ),
    phoneme_cache_path=phoneme_cache_path,
    compute_input_seq_cache=True,
    print_step=args.print_step,
    print_eval=False,
    mixed_precision=True,
    grad_clip=[5., 5.],
    datasets=[pretraining_dataset_config, user_dataset_config],
    dashboard_logger=args.logger,
    logger_uri=tensorboard_dir,
    test_sentences=[
        ["It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.", speaker, ""],
        ["Be a voice, not an echo.", speaker, ""],
        ["I'm sorry Dave. I'm afraid I can't do that.", speaker, ""],
        ["This cake is great. It's so delicious and moist.", speaker, ""],
        ["Prior to November 22, 1963.", speaker, ""],
    ],
    use_weighted_sampler=True,
    weighted_sampler_attrs={"speaker_name":1.},
    weighted_sampler_multipliers={"speaker_name":{args.tgt_speaker:10}} #TODO(arto): heuristic for upweighting tgt speaker based on amount of data
)

ap = AudioProcessor.init_from_config(config)
tokenizer, config = TTSTokenizer.init_from_config(config)
train_samples, eval_samples = load_tts_samples(
    [pretraining_dataset_config, user_dataset_config],
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

create_speakers_file(model_dir, [speaker], args.parent_model_dir)

speaker_manager = SpeakerManager()
speaker_manager.load_ids_from_file(os.path.join(model_dir, 'speakers.json'))
config.model_args.num_speakers = speaker_manager.num_speakers
# init model
model = Vits(config, ap, tokenizer, speaker_manager=speaker_manager)

#freeze text encoder
# for n, p in model.named_parameters():
#     if n.startswith('text_encoder'):
#         p.requires_grad = False

trainer = Trainer(
    args=TrainerArgs(restore_path=args.restore_path, continue_path=continue_path),
    config=config,
    output_path=checkpoint_dir,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()


try:
    best_model_path = sorted(list(Path(checkpoint_dir).rglob('best_model.pth')), key=os.path.getmtime)[-1]
    repackage_model(best_model_path.as_posix(), os.path.join(model_dir, 'model_file.pth'))
    config_path = best_model_path.with_name('config.json')
    create_infrence_config(config_path.as_posix(), os.path.join(model_dir, "config_inference_gcp.json"))

    r = requests.post(
        f'{parent_api_url}/v1/trainingstatus', 
        json={"uuid":task_id, "status":"completed"},
        headers={'x-api-key':x_api_key}
    )
    log_task_status_upd(r.status_code)
except:
    print("!!Best model not found. Failed to export model artifacts!!")

    r = requests.post(
        f'{parent_api_url}/v1/trainingstatus', 
        json={"uuid":task_id, "status":"failed"},
        headers={'x-api-key':x_api_key}
    )
    log_task_status_upd(r.status_code)
    