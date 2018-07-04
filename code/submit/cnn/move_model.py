import os

files = os.listdir('./snapshot/')
print files
files = sorted(files, key=lambda x:float(x.split('_')[3][:-3]), reverse=True)
print files[0]
