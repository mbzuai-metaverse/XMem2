import re
import os 
def get_numbers_from_filename(filename):
    return re.search(r'\d+', filename).group(0)

lst = []
for filename in os.listdir('/home/ariana/RESEARCH/Datasets/metaverse/blog/Annotations/Together'):
   lst += [int(get_numbers_from_filename(filename))]
print(sorted(lst))
print(len(lst))