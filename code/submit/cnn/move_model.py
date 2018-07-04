import os

files = os.listdir('./snapshot/')
files = sorted(files, key=lambda x:float(x.split('_')[3][:-3]), reverse=True)
os.system('cp ' + files[0] + 'cv_models/')
os.system('rm snapshot/*.pt')