from typing import Tuple, Dict, Union, Any, Type

REQUIRED: tuple[str, ...] = ('no voice', 'no voices', 'no vocal', 'no vocals', 'instrumental')
TAGS: tuple[str, ...] = ('X',
                         'no voice',
                         'singer',
                         'duet',
                         'plucking',
                         'hard rock',
                         'world',
                         'bongos',
                         'harpsichord',
                         'female singing',
                         'classical',
                         'sitar',
                         'chorus',
                         'female opera',
                         'male vocal',
                         'vocals',
                         'clarinet',
                         'heavy',
                         'silence',
                         'beats',
                         'men',
                         'woodwind',
                         'funky',
                         'no strings',
                         'chimes',
                         'foreign',
                         'no piano',
                         'horns',
                         'classical',
                         'female',
                         'no voices',
                         'soft rock',
                         'eerie',
                         'spacey',
                         'jazz',
                         'guitar',
                         'quiet',
                         'no beat',
                         'banjo',
                         'electric',
                         'solo',
                         'violins',
                         'folk',
                         'female voice',
                         'wind',
                         'happy',
                         'ambient',
                         'new age',
                         'synth',
                         'funk',
                         'no singing',
                         'middle eastern',
                         'trumpet',
                         'percussion',
                         'drum',
                         'airy',
                         'voice',
                         'repetitive',
                         'birds',
                         'space',
                         'strings',
                         'bass',
                         'harpsicord',
                         'medieval',
                         'male voice',
                         'girl',
                         'keyboard',
                         'acoustic',
                         'loud',
                         'classic',
                         'string',
                         'drums',
                         'electronic',
                         'not classical',
                         'chanting',
                         'no violin',
                         'not rock',
                         'no guitar',
                         'organ',
                         'no vocal',
                         'talking',
                         'choral',
                         'weird',
                         'opera',
                         'soprano',
                         'fast',
                         'acoustic guitar',
                         'electric guitar',
                         'male singer',
                         'man singing',
                         'classical guitar',
                         'country',
                         'violin',
                         'electro',
                         'reggae',
                         'tribal',
                         'dark',
                         'male opera',
                         'no vocals',
                         'irish',
                         'electronica',
                         'horn',
                         'operatic',
                         'arabic',
                         'lol',
                         'low',
                         'instrumental',
                         'trance',
                         'chant',
                         'strange',
                         'drone',
                         'synthesizer',
                         'heavy metal',
                         'modern',
                         'disco',
                         'bells',
                         'man',
                         'deep',
                         'fast beat',
                         'industrial',
                         'hard',
                         'harp',
                         'no flute',
                         'jungle',
                         'pop',
                         'lute',
                         'female vocal',
                         'oboe',
                         'mellow',
                         'orchestral',
                         'viola',
                         'light',
                         'echo',
                         'piano',
                         'celtic',
                         'male vocals',
                         'orchestra',
                         'eastern',
                         'old',
                         'flutes',
                         'punk',
                         'spanish',
                         'sad',
                         'sax',
                         'slow',
                         'male',
                         'blues',
                         'vocal',
                         'indian',
                         'no singer',
                         'scary',
                         'india',
                         'woman',
                         'woman singing',
                         'rock',
                         'dance',
                         'piano solo',
                         'guitars',
                         'no drums',
                         'jazzy',
                         'singing',
                         'cello',
                         'calm',
                         'female vocals',
                         'voices',
                         'different',
                         'techno',
                         'clapping',
                         'house',
                         'monks',
                         'flute',
                         'not opera',
                         'not english',
                         'oriental',
                         'beat',
                         'upbeat',
                         'soft',
                         'noise',
                         'choir',
                         'female singer',
                         'rap',
                         'metal',
                         'hip hop',
                         'quick',
                         'water',
                         'baroque',
                         'women',
                         'fiddle',
                         'english',
                         'X')

TAG_COUNT = len(TAGS) - 2

TAG_MAP: dict[str, str] = {'rock': 'rock',
                           'hard rock': 'rock',
                           'metal': 'rock',
                           'heavy metal': 'rock',
                           'orchestra': 'orchestral',
                           'classical': 'orchestral',
                           'techno': 'electronic',
                           'ambient': 'electronic',
                           'house': 'electronic',
                           'hip hop': 'hip hop',
                           'rap': 'hip hop',
                           'calm': 'test/safe',
                           'happy': 'test/safe',
                           'mellow': 'test/safe',
                           'scary': 'test/dangerous',
                           'eerie': 'test/dangerous',
                           'fast beat': 'test/dangerous',
                           'jazz': 'jazz'}

JSON_FIELDS: tuple[str, ...] = ('id', 'tags', 'file')

JSON_MFCC_PATH: str = '../resources/fma/mfcc.json'
JSON_ONSET_PATH: str = '../resources/fma/onset.json'
FMA_FEATURES_PATH: str = '../resources/fma/features.json'
FMA_BUCKETS_PATH: str = '../resources/fma/buckets.json'

GTZ_FEATURES_PATH: str = '../resources/gtzan/features.json'
GTZ_BUCKETS_PATH: str = '../resources/gtzan/buckets.json'


ENERGY_FIELD: str = 'Energy'
MFCC_FIELD: str = 'MFCCS_BUCKET'
MFCC_FIELDS: tuple[str, ...] = ('Name', 'Index', 'Min', 'Max', 'Energy', 'BUCKET')
ONSET_FIELDS: tuple[str, ...] = ('Name', 'Index', 'Onsets', 'BUCKET')
FEATURES_FIELDS: tuple[str, ...] = ("Onsets", "mfcc_mean", "melSpec_mean", "chromaVec_mean", "roll_mean", "zcr_mean")
FEATURES_FIELDS_SMALL: tuple[str, ...] = ("Onsets", "mfcc_mean")
OUTPUT_BUCKETS: tuple[str, ...] = ("00", "01", "02", "03", "10", "11", "12", "13", "20", "21", "22", "23", "30", "31",
                                   "32", "33")

###############################
#            STUDY            #
###############################

TRACKS_ARR: tuple[str, ...] = (
    '000740', '000756', '001554', '001730', '003497', '003499', '004066', '004069', '004511', '004703',
    '004850', '005222', '005224', '005348', '006525', '006658', '006664', '006778', '007545', '007709',
    '009512', '009989', '010442', '010690', '010737', '011201', '011432', '011454', '011612', '011724',
    '011748', '011752', '012489', '013014', '013578', '013582', '013814', '013864', '014053', '014216',
    '014390', '014921', '015094', '015145', '016057', '016095', '016439', '016448', '016554', '016878',
    '016997', '017324', '017344', '017609', '017910', '018022', '018578', '018579', '019266', '019354',
    '019355', '020376', '020432', '020449', '021542', '021708', '021848', '021898', '022128', '023013',
    '023146', '023525', '023820', '024178', '024418', '024422', '024429', '024640', '024794', '024995',
    '025005', '025145', '025364', '025804', '026171', '026630', '026632', '026697', '027148', '027633',
    '027752', '027866', '028111', '028175', '028178', '028476', '029324', '029469', '029957', '030071',
    '030095', '030740', '031234', '031236', '031574', '031576', '032326', '032517', '033064', '033220',
    '033271', '034513', '035299', '035454', '035461', '035539', '035569', '035991', '036144', '036187',
    '036243', '036485', '036988', '036990', '038318', '038361', '038679', '038819', '039282', '039530',
    '039941', '039947', '039949', '039984', '040162', '040175', '040369', '040782', '040784', '040798',
    '041090', '041453', '041870', '041939', '042234', '042286', '042756', '042842', '043509', '044344',
    '044359', '044821', '044985', '045012', '046144', '046147', '047016', '047481', '047832', '047835',
    '048412', '048453', '048731', '049361', '050167', '051657', '051785', '052004', '052358', '052404',
    '052638', '052648', '054004', '054062', '054124', '054131', '054137', '054465', '054833', '055100',
    '055109', '055821', '056273', '056649', '056802', '056872', '056882', '057180', '058130', '058757',
    '059173', '059192', '059504', '059679', '060510', '060701', '060757', '062049', '064445', '064447',
    '064450', '064542')

EVAL_KEYS_A: tuple[str, ...] = ('Danger', 'Urgency', 'Risk of Failure', 'Collaboration', 'Approachable')

EVAL_KEYS_B: tuple[str, ...] = ('Safety', 'Urgency', 'Risk of Failure', 'Collaboration', 'Approachable')

SELECTION_FIELD: str = 'selected_mushra'

PEOPLE_FIELDS: tuple[str, ...] = ('UserLanguage', 'LocationLatitude', 'LocationLongitude', 'Duration (in seconds)')

PEOPLE_TYPES: dict[str, Any] = {'LocationLatitude': float, 'LocationLongitude': float, 'Duration (in seconds)': int}

STUDY_PARTITION = 368

RATING_RANGE: range = range(18, 717, 5)

ID_FIELD: str = 'ResponseId'

START_ROW: int = 1

################################
#      FEATURE EXTRACTION      #
################################

SAMPLE_RATE = 22050


def void(*args):
    return args


if __name__ == '__main__':
    print(STUDY_PARTITION in RATING_RANGE)
    print(len(TRACKS_ARR))
