import pvlib
CECMODS = pvlib.pvsystem.retrieve_sam(name='CECMod')
test = CECMODS.T['Technology']
i = 0
j = 0
for t in test:
    if "a" in t :#and i< 1980:
        bob = t
        bill = i
        print(bob,bill)
        j = j+1
    i = i+1
