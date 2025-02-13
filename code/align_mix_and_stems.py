import soundfile as sf
import os; opj = os.path.join
from pprint import pprint
from glob import glob
from tqdm import tqdm
import numpy as np
import parmap
import audalign as ad
import pickle
import librosa
import shutil

def get_rough_sum(song):
    dry_tracks = sorted(glob(opj(song, "*/*.wav")))
    dry_tracks = [d for d in dry_tracks if not "mix" in d.lower()]
    drys = []
    for dry_track in dry_tracks:
        wav, sr = sf.read(dry_track, always_2d=True)
        if sr != 44100:
            wav = librosa.resample(wav.T, orig_sr=sr, target_sr=44100).T
        drys.append(wav)
    if len(drys) != 0:
        max_len = max([len(x) for x in drys])
        drys = [np.pad(x, ((0, max_len-len(x)), (0, 0))) for x in drys]
        rough_sum = sum(drys)
        sf.write(opj(song, "rough_mix.wav"), rough_sum, 44100)

def align_song(song):
    try:
        recognizer = ad.CorrelationSpectrogramRecognizer()
        fine_recognizer = ad.CorrelationRecognizer()
        fine_recognizer.config.sample_rate = 44100
        mix_dir = opj(song, "mix.wav")
        dry_dir = opj(song, "rough_mix.wav")
        results = ad.align_files(
                mix_dir, 
                dry_dir,
                #destination_path=align_dir,
                recognizer=recognizer
                )
        fine_recognizer.config.max_lags = 0.05
        results = ad.fine_align(
                results=results,
                recognizer=fine_recognizer,
                )
        pickle.dump(results, open(opj(song, "alignment.pickle"), "wb"))
    except:
        return

def align_songs():
    songs = sorted(glob("44k_pv_only/*/"))
    SPLITS = 8
    split = 7
    num_songs = int(np.ceil(len(songs)/SPLITS))
    for i in tqdm(range(
        num_songs*split, 
        min(num_songs*(split+1), len(songs))
        )):
        align_song(songs[i])

def rms(x):
    return np.sqrt(np.mean(np.square(x)))

def align_and_save(song):
    try:
        align_data = pickle.load(open(opj(song, "alignment.pickle"), "rb"))
        offset = align_data["mix.wav"]-align_data["rough_mix.wav"]
            #align_song(song)
            #print(align_data)
        mix, rough_mix = sf.read(opj(song, "mix.wav"), always_2d=True)[0], sf.read(opj(song, "rough_mix.wav"), always_2d=True)[0]
        mix, rough_mix = np.mean(mix, -1), np.mean(rough_mix, -1)
        offset_sample = int(round(offset*44100))
        if offset_sample > 0:
            rough_mix = rough_mix[offset_sample:]
        else:
            rough_mix = np.pad(rough_mix, (-offset_sample, 0))
            
        mix = mix[:len(rough_mix)]
        rough_mix = rough_mix[:len(mix)]
        rough_mix = rough_mix/rms(rough_mix)*rms(mix)

        comp = np.stack([mix, rough_mix], -1)
        sf.write(opj(song, "comp.wav"), comp, 44100)
        sf.write(opj("align_comp_pv", f"{song[12:-1]}.wav"), comp, 44100)
    except:
        return

def update_meta():
    songs = glob("44k_pv_only/*")
    for song in songs:
        try:
            src, dst = opj(song, "correspondence.yaml"), opj("30k_extra", os.path.basename(song), "correspondence.yaml")
            shutil.copy(src, dst)
            src, dst = opj(song, "alignment.pickle"), opj("30k_extra", os.path.basename(song), "alignment.pickle")
            shutil.copy(src, dst)
        except:
            continue

if __name__ == "__main__":
    #songs = glob("44k_pv_only/*/")
    #parmap.map(get_rough_sum, songs, pm_processes=16, pm_pbar=True)
    #parmap.map(align_song, songs, pm_processes=16, pm_pbar=True)
    #parmap.map(align_and_save, songs, pm_processes=16, pm_pbar=True)
    #align_songs()
    #check_align()
    update_meta()
