import getpass

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


