import argparse
from pathlib import Path
import torch
import pandas as pd

from TTS.utils.synthesizer import Synthesizer



parser = argparse.ArgumentParser(
	    description='Synthesize speech from text script')
parser.add_argument("-m", "--model", type=str,
                    required=True, help="Model path")
parser.add_argument("-c", "--config", type=str,
                    required=True, help="Config path")
parser.add_argument("-vm", "--vocoder_model", type=str,
                    default=None, help="Vocoder model path")
parser.add_argument("-vc", "--vocoder_config", type=str,
                    default=None, help="Vocoder config path")
#parser.add_argument("-sp", "--speakers_file", type=str,
#                    required=True, help="Speakers file path")
parser.add_argument("-d", "--data", type=str,
					 required=True, help="csv with text to extract")
args = parser.parse_args()

model_path = Path(args.model)
config_path = Path(args.config)
vocoder_path = Path(args.vocoder_model) if args.vocoder_model else None
vocoder_config_path = Path(args.vocoder_config) if args.vocoder_config else None
ds_path = Path(args.data)
#speakers_file_path = Path(args.speakers_file)
#out_file = Path(args.output)


use_cuda = True if torch.cuda.is_available() else False

speakers_file_path = None
encoder_path = None
encoder_config_path = None

# load models
synthesizer = Synthesizer(
    model_path,
    config_path,
    speakers_file_path,
    vocoder_path,
    vocoder_config_path,
    encoder_path,
    encoder_config_path,
    use_cuda,
)

df = pd.read_csv(ds_path, usecols=["ID", "wrd"])
out_dir = Path("/local_disk/calypso/jduret/corpus/NER/quaero") / ds_path.stem
out_dir.mkdir(exist_ok=True)

for id, text in zip(df['ID'], df['wrd']):
	print(" > Text: {}".format(text))
	wav = synthesizer.tts(text)
	out_file = out_dir / f"{id}.wav" 
	print(" > Saving output to {}".format(out_file))
	synthesizer.save_wav(wav, out_file)
