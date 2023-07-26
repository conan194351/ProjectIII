import subprocess
import sys
import os
# NEW ACC
lrdecay = str(0.975)


# print ("CRV1=====================Training and Test")
# # subprocess.check_call([sys.executable, 'Training_kFold_CNN_VoiceID.py','--fold',"CRV1",'--lr_decay',lrdecay])
# subprocess.check_call([sys.executable, 'Test_CNN_VoiceID.py','--fold',"CRV1",'--lr_decay',lrdecay])
# #
print ("CRV2=====================Training and Test")
subprocess.check_call([sys.executable, 'Training.py','--fold',"CRV2",'--lr_decay',lrdecay])
# subprocess.check_call([sys.executable, 'Test_CNN_VoiceID.py','--fold',"CRV2",'--lr_decay',lrdecay])

# print ("CRV3=====================Training and Test")
# subprocess.check_call([sys.executable, 'Training_kFold_CNN_VoiceID.py','--fold',"CRV3",'--lr_decay',lrdecay])
# subprocess.check_call([sys.executable, 'Test_CNN_VoiceID.py','--fold',"CRV3",'--lr_decay',lrdecay])

# print ("CRV4=====================Training and Test")
# subprocess.check_call([sys.executable, 'Training_kFold_CNN_VoiceID.py','--fold',"CRV4",'--lr_decay',lrdecay])
# subprocess.check_call([sys.executable, 'Test_CNN_VoiceID.py','--fold',"CRV4",'--lr_decay',lrdecay])

# print ("CRV5=====================Training and Test")
# subprocess.check_call([sys.executable, 'Training_kFold_CNN_VoiceID.py','--fold',"CRV5",'--lr_decay',lrdecay])
# subprocess.check_call([sys.executable, 'Test_CNN_VoiceID.py','--fold',"CRV5",'--lr_decay',lrdecay])

# print ("CRV6=====================Training and Test")
# subprocess.check_call([sys.executable, 'Training_kFold_CNN_VoiceID.py','--fold',"CRV6",'--lr_decay',lrdecay])
# subprocess.check_call([sys.executable, 'Test_CNN_VoiceID.py','--fold',"CRV6",'--lr_decay',lrdecay])

# print ("CRV7=====================Training and Test")
# subprocess.check_call([sys.executable, 'Training_kFold_CNN_VoiceID.py','--fold',"CRV7",'--lr_decay',lrdecay])
# subprocess.check_call([sys.executable, 'Test_CNN_VoiceID.py','--fold',"CRV7",'--lr_decay',lrdecay])

# print ("CRV8=====================Training and Test")
# subprocess.check_call([sys.executable, 'Training_kFold_CNN_VoiceID.py','--fold',"CRV8",'--lr_decay',lrdecay])
# subprocess.check_call([sys.executable, 'Test_CNN_VoiceID.py','--fold',"CRV8",'--lr_decay',lrdecay])

# print ("CRV9=====================Training and Test")
# subprocess.check_call([sys.executable, 'Training_kFold_CNN_VoiceID.py','--fold',"CRV9",'--lr_decay',lrdecay])
# subprocess.check_call([sys.executable, 'Test_CNN_VoiceID.py','--fold',"CRV9",'--lr_decay',lrdecay])