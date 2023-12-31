from PyInstaller.utils.hooks import collect_data_files, get_package_paths
import os
import re 

hiddenimports = ['_cffi_backend']

# Get the package path
package_path = get_package_paths('coincurve')[0]

# Collect data files
datas = collect_data_files('coincurve')

# Regular expression pattern for libsecp256k1
libsecp256k1_pattern = re.compile(r'(libsecp256k1|_libsecp256k1).*\.(dll|so)')

# Search for libsecp256k1 file within the 'coincurve' folder
coincurve_path = os.path.join(package_path, 'coincurve')
libsecp256k1_path = None
for root, _, files in os.walk(coincurve_path):
    for file in files:
        if libsecp256k1_pattern.search(file):
            libsecp256k1_path = os.path.join(root, file)
            datas.append((libsecp256k1_path, 'coincurve'))
            break

    assert libsecp256k1_path, f"can't find coincurve libsecp256k1, look in {coincurve_path} and then modify this hook"
