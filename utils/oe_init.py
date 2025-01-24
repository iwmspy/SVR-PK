import os
from openeye.oechem import *
homedir = os.environ['HOME']
OESetLicenseFile(os.path.join(homedir, '.OpenEye', 'oe_license.txt'))
if OEChemIsLicensed():
    print('OpenEye certification succeeded!')
else:
    print('OpenEye certification failed...')