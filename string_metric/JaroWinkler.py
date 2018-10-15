import pyjarowinkler

from pyjarowinkler import distance

def metrics(str1, str2):
    return distance.get_jaro_distance(str1, str2, winkler=True, scaling=0.1)

