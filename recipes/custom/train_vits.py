import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, CharactersConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.speakers import SpeakerManager

output_path = "/home/freeman-vits-vctk"
dataset_config = BaseDatasetConfig(
    name="vctk_freeman", meta_file_train='train', path="/home/data", meta_file_val='dev'
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

config = VitsConfig(
    model_args=model_args,
    audio=audio_config,
    run_name="vits-vctk-freeman",
    run_description="Fine-tune VITS on VCTK with added Freeman dataset",
    project_name="voicemod",
    wandb_entity="arampacha",
    batch_size=48,
    eval_batch_size=16,
    batch_group_size=5,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=0,
    epochs=500,
    save_step=1000,
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
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=100,
    print_eval=False,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    dashboard_logger='wandb',
    test_sentences=[
        ["It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.", "Freeman_angry"],
        ["Be a voice, not an echo.", "Freeman_normal"],
        ["I'm sorry Dave. I'm afraid I can't do that.", "Freeman_narration"],
        ["This cake is great. It's so delicious and moist.", "Freeman_happy"],
        ["Prior to November 22, 1963.", "Freeman_narration"],
    ],
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

speaker_manager = SpeakerManager()
speaker_manager.load_ids_from_file(os.path.join(output_path, 'speakers.json'))
config.model_args.num_speakers = speaker_manager.num_speakers
# init model
model = Vits(config, ap, tokenizer, speaker_manager=speaker_manager)

#freeze text encoder
# for n, p in model.named_parameters():
#     if n.startswith('text_encoder'):
#         p.requires_grad = False

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()