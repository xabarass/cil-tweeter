import getpass
import time
import os

user_name = getpass.getuser()

# Configuration data
if user_name in {"milan","pmilan"}:
    from .milan_config import *
elif user_name in {"franzm"}:
    from .michael_config import *
elif user_name in {"allandre"}:
    from .andreas_config import *
elif user_name in {"lukas","lukasd"}:
    from .lukas_config import *
elif user_name in {"nforster"}:
    from .azure_config import *



timestamp = str(int(time.time()))

def _make_file_path(path):
    file_path = 'runs/' + timestamp + '/' + path
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    return file_path

word_embeddings_opt["corpus_name"] = _make_file_path( word_embeddings_opt["corpus_name"] )

result_epoch_file =                  _make_file_path( ('-e{}_' + timestamp + '.').join( result_file.split('.') ) )
result_file =                        _make_file_path( ('_' + timestamp + '.').join( result_file.split('.') ) )

model_save_path =                    _make_file_path(model_save_path)
misclassified_samples_file =         _make_file_path(misclassified_samples_file)
