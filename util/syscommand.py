import subprocess
import sys


def system(command: str) -> str:
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    stdout, stderr = process.communicate()
    return stdout.decode(sys.stdout.encoding)
