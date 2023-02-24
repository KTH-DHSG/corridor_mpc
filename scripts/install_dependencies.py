#!/usr/bin/env python
import platform
import os
import subprocess

if __name__ == '__main__':
    if 'ubuntu' in platform.version().lower():
        proc = subprocess.Popen(["lsb_release -r"], stdout=subprocess.PIPE, shell=True)
        (rls, err) = proc.communicate()
        version = float(rls.decode('UTF-8').split()[1])

        if version == 20.04 or version == 22.04 or version == 18.04:
            print("[Plotting Dependency] Installing cm-super package...")
            os.system('sudo apt install cm-super')
    else:
        print("Please install the 'cm-super' package manually.")
        exit()
