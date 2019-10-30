import os
import sys

res = os.path.dirname(os.path.realpath(sys.argv[0]))
print(res)
print(sys.argv[0])
file_name =  os.path.basename(sys.argv[0])
print(file_name)
