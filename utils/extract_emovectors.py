import argparse
import glob
import json 
import pathlib as pl

import torchaudio
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        required=True,
        help="Path of the folder containing the audio files",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Path of the folder containing the embeddings file",
    )

    args = parser.parse_args()

    input_dir = pl.Path(args.input_dir)
    output_dir = pl.Path(args.output_dir)

    audio_files = glob.glob(str(input_dir / "**/*_mic2.flac"), recursive=True)
    print(f"Found {len(audio_files)} files...")

    spk_embs = {}
    classifier = EncoderClassifier.from_hparams(source="/local_disk/calypso/jduret/git/Chaanks/sb-interfaces/emotion_v2", run_opts={"device":"cuda"})
    
    for audio in tqdm(audio_files):
        spk_name, filename = audio.split('/')[-2:]
        signal, fs = torchaudio.load(audio_files[0])
        embedding = classifier.encode_batch(signal)
        embedding = embedding.detach().cpu().numpy().squeeze()
        embedding = [float(x) for x in embedding]
        
        spk_embs[filename] = {
            "name": spk_name,
            "embedding": embedding
        }
    
    with open(output_dir / "speakers_emovectors.json", 'w') as f:
        json.dump(spk_embs, f)