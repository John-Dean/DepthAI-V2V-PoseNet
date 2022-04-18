import subprocess
import sys
from os.path import exists

# https://stackoverflow.com/a/58026969/5494277
in_venv = getattr(sys, "real_prefix", getattr(sys, "base_prefix", sys.prefix)) != sys.prefix
pip_call = [sys.executable, "-m", "pip"]
pip_install = pip_call + ["install"]

if not in_venv:
    pip_install.append("--user")

subprocess.check_call([*pip_install, "pip", "-U"])
subprocess.check_call([*pip_install, "-r", "requirements.txt"])


if exists("requirements-optional.txt"):
    try:
        subprocess.check_call([*pip_install, "-r", "requirements-optional.txt"], stderr=subprocess.STDOUT)
    except:
        print(f"Optional dependencies were not installed")
