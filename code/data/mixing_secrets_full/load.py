import pickle
from functools import partial
from glob import glob
from os.path import dirname, isfile, join, realpath, basename

import numpy as np
import soundfile as sf
from utils import flatten
from yaml import safe_load

script_path = dirname(realpath(__file__))
base_config_dir = join(script_path, "configs.yaml")
configs = safe_load(open(base_config_dir, "rb"))

BASE_DIR = configs["base_dir"]
SKIPS = configs["skips"]
# print("SKIPS", SKIPS)

def detach_base_dir(x, d=4):
    return join(*x.split("/")[-d:])


def check_song(song, min_num_inputs=0, max_num_inputs=150):
    if any([s in song for s in SKIPS]):
        return False
    # print("song", basename(song))
    # check if the song has already been processed
    log_dir = "/data4/soumya/workspace/grafx-prune-new"
    if isfile(join(log_dir, f"mixing_secrets_full_{basename(song)}","prune_hybrid_1e_2_result.pickle" )):
        return False

    # print("checking song", song)
    if not isfile(join(song, "aligned/full_alignment.pickle")):
        # print("no full_alignment.pickle")
        return False
    if not isfile(join(song, "aligned/correspondance_full.yaml")):
        # print("no correspondence_full.yaml")
        return False
    metadata = safe_load(open(join(song, "aligned/correspondance_full.yaml"), "r"))
    num_dry = len(flatten(list(metadata.values())))
    
    if num_dry < min_num_inputs:
        return False
    if num_dry > max_num_inputs:
        return False
    return True


def get_mixing_secrets_full_song_list(
    mode, seed=0, n_valid=24, n_test=24, min_num_inputs=0, max_num_inputs=150
):

    song_list = sorted(glob(join(BASE_DIR, "*")))
    print(f"found  {len(song_list)} songs")
    # remove items from the list that dont contain a specific file
    song_list = [s for s in song_list if isfile(join(s, "full_mix_previews/full_mix_preview.mp3"))]
    print(f"found  {len(song_list)} songs")

    filter_func = partial(
        check_song, min_num_inputs=min_num_inputs, max_num_inputs=max_num_inputs
    )
    song_list = list(filter(filter_func, song_list))

    num_songs = len(song_list)
    print(f"found  {num_songs} songs after all the filtering")
    assert num_songs != 0
    n_train = num_songs - n_valid - n_test
    n_valid = n_train + n_valid

    rng = np.random.RandomState(seed)
    rng.shuffle(song_list)
    print(mode)
    match mode:
        case "train":
            song_list = sorted(song_list[:n_train])
        case "valid":
            song_list = sorted(song_list[n_train:n_valid])
        case "test":
            song_list = sorted(song_list[n_valid:])
        case "all":
            song_list = sorted(song_list)
        case _:
            assert False

    assert len(song_list) != 0
    song_list = [detach_base_dir(s, 1) for s in song_list]
    return song_list


def load_mixing_secrets_full_metadata(song, sr=30000):
    metadata = {}
    # song = "'" + song + "'"
    # print("song", song)
    metadata["song"] = song
    metadata["dataset"] = "mixing_secrets_full"
    metadata["base"] = BASE_DIR
    print("BASE_DIR", BASE_DIR)
    print("song", song)
    song_dir = join(BASE_DIR, song)
    metadata["song_dir"] = song_dir

    raw_metadata_dir = join(song_dir, "aligned", f"correspondance_full.yaml")
    if isfile(raw_metadata_dir):
        raw_metadata = safe_load(open(raw_metadata_dir, "r"))
    else:
        print("?", raw_metadata_dir)
    correspondence_data = dict(matched=raw_metadata)
    metadata["correspondence"] = correspondence_data

    matched_dry_dirs = flatten(list(correspondence_data["matched"].values()))
    # print("matched_dry_dir", matched_dry_dirs)
    dry_dir = glob(join(song_dir, "full_multitrack", "*", "*.wav"))[0]
    dry_dir = dirname(dry_dir)
    # print("dry_dir",dry_dir)
    metadata["matched_dry_dirs"] = [join(dry_dir, d) for d in matched_dry_dirs]
    # print("metadata['matched_dry_dirs']",metadata["matched_dry_dirs"])  
    metadata["mix_dir"] = join(song_dir, "full_mix_previews/full_mix_preview.mp3")
    metadata["total_len"] = sf.info(metadata["mix_dir"]).frames

    alignment_dir = join(song_dir, "aligned", f"full_alignment.pickle")
    alignment_data = pickle.load(open(alignment_dir, "rb"))
    offset = alignment_data["full_mix_preview.mp3"] - alignment_data["full_rough_mix.wav"]
    offset_sample = int(round(offset * sr))
    metadata["dry_alignment"] = -offset_sample

    return metadata
