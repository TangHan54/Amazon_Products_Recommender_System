import os

items = os.listdir("data")

for names in items:
    if names.endswith(".gz"):
        names = names.rstrip('.gz')
        if names not in items:
            print(f'gunzip data/{names}.gz')
            os.system(f'gunzip data/{names}.gz')

        


