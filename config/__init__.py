import getpass

user_name = getpass.getuser()

# Configuration data
if user_name in set("milan","pmilan"):
    from .milan_config import *
elif user_name in set("franzm"):
    from .michael_config import *
elif user_name in set("allandre"):
    from .andreas_config import *
elif user_name in set("lukas","lukasd"):
    from .lukas_config import *