import pandas as pd

class IdTranslator(object):
    """
    Class for converting song ids to track ids
    """
    def __init__(self, translation_file):
        self.tl = pd.read_csv(translation_file, sep='\t',
                              names=['Song ID', 'Track ID'])

    def get_track_id(self, song_id):
        song_ids = self.tl.get('Song ID')
        idx = song_ids[song_ids == song_id].index[0]
        return self.tl.get('Track ID')[idx]

    def get_song_id(self, track_id):
        track_ids = self.tl.get('Track ID')
        idx = track_ids[track_ids == track_id].index[0]
        return self.tl.get('Song ID')[idx]


