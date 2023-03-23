from distutils.sysconfig import get_python_inc
import sys
header_path = get_python_inc()

print(header_path, file=sys.stdout)