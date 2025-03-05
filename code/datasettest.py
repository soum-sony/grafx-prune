

from data.dataset import SingleTrackDataset
from data.load import get_song_list
import argparse

if __name__=="__main__":

# load args yaml from config
    import yaml
    with open("configs/base.yaml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    # dataset = config["datasets"][0]
    # song = config["song"]
    # mode = config["dataset_split"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=config["datasets"])
    parser.add_argument("--song", default=config["song"])
    parser.add_argument("--dataset_split", default=config["dataset_split"])
    args = parser.parse_args()
    song_list = []
    for dataset in args.datasets:
        l = get_song_list(mode=args.dataset_split, dataset=dataset)
        song_list += [(dataset, song) for song in l]
        print("song_list", song_list)

    for (datasetname, songname) in song_list:
        loaddataset = SingleTrackDataset(mode = args.dataset_split, dataset = datasetname, song = songname)
        getitem = loaddataset.__getitem__(0)
        print(getitem)
        print(loaddataset.__len__())
    