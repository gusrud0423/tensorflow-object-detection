import tensorflow as tf 
# print(tf.__version__)  # >> 2.4.1 버전이다

import tarfile  
import urllib.request
import os

### 네트워크 통해 모델 다운로드 하고 압축 푸는 코드 ### 순서대로
MODEL_DATE = '20200711'
MODEL_NAME = 'ssd_mobilenet_v2_320x320_coco17_tpu-8'
MODEL_TAR_FILENAME = MODEL_NAME + '.tar.gz'

MODELS_DIR = 'data/models'

# 모델 다운로드 받을 수 있는 곳 
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
MODELS_DOWNLOAD_BASE =  'http://download.tensorflow.org/models/object_detection/tf2/'
MODEL_DOWNLOAD_LINK = MODELS_DOWNLOAD_BASE + MODEL_DATE + '/' + MODEL_TAR_FILENAME

PATH_TO_MODEL_TAR = os.path.join( 'data/models', MODEL_TAR_FILENAME )
PATH_TO_CKPT = os.path.join( 'data/models', os.path.join( MODEL_NAME, 'checkpoint' ) )
PATH_TO_CFG = os.path.join( 'data/models', os.path.join( MODEL_NAME, 'pipeline.config' ) )

# 모델을 받아서 압축풀기 
if not os.path.exists(PATH_TO_CKPT):
    print('Downloading model. This may take a while... ', end='')
    urllib.request.urlretrieve(MODEL_DOWNLOAD_LINK, PATH_TO_MODEL_TAR)
    tar_file = tarfile.open(PATH_TO_MODEL_TAR)
    tar_file.extractall(MODELS_DIR)
    tar_file.close()
    os.remove(PATH_TO_MODEL_TAR)
    print('Done')

# 레이블 다운로드 받기 
LABEL_FILENAME = 'mscoco_label_map.pbtxt'
LABELS_DOWNLOAD_BASE = \
    'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
PATH_TO_LABELS = os.path.join(MODELS_DIR, os.path.join(MODEL_NAME, LABEL_FILENAME))
if not os.path.exists(PATH_TO_LABELS):
    print('Downloading label file... ', end='')
    urllib.request.urlretrieve(LABELS_DOWNLOAD_BASE + LABEL_FILENAME, PATH_TO_LABELS)
    print('Done')

####### 이것을 실행 시킨 결과 >> 목록에 생성됨 
# Downloading model. This may take a while... Done
# Downloading label file... Done


### 모델 로딩 ! ###
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

# GPU의 Dynamic Memory Allocation 을 활성화 시키는 코드 
gpus = tf.config.experimental.list_physical_devices('GPU')  # GPU 를 가져오는 코드 ~
for gpu in gpus :
    tf.config.experimental.set_memory_growth( gpu, True )

# config 로드하고, 모델 빌드 
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build( model_config= model_config, is_training= False )

# Restore Checkpoint
ckpt = tf.compat.v2.train.Checkpoint( model = detection_model )
ckpt.restore( os.path.join( PATH_TO_CKPT, 'ckpt-0' ) ).expect_partial()

@tf.function
def detect_fn(image) :
    """Detect Objects in Image."""
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections =  detection_model.postprocess( prediction_dict, shapes )

    return detections, prediction_dict, tf.reshape( shapes, [-1] )

# 레이블 맵 데이터 로딩 ( Load label map data )
category_index =  label_map_util.create_categories_from_labelmap( PATH_TO_LABELS, use_display_name = True )
print(category_index)

## 비디오 실행해보기 ?
import c