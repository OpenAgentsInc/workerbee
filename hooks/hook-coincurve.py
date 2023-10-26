from PyInstaller.utils.hooks import collect_data_files, get_package_paths
import os

hiddenimports=['_cffi_backend']

# Get the package path
package_path = get_package_paths('coincurve')[0]

# Collect data files
datas = collect_data_files('coincurve')

# Append the additional .dll or .so file
if os.name == 'nt':  # Windows
    dll_path = os.path.join(package_path, 'coincurve', 'libsecp256k1.dll')
    datas.append((dll_path, 'coincurve'))
elif os.name == 'posix':  # Linux/Mac
    so_paths =[
            os.path.join(package_path, 'coincurve', 'libsecp256k1.so'),
            os.path.join(package_path, 'coincurve', '_libsecp256k1.cpython-311-x86_64-linux-gnu.so')
    ]
    so_path = None
    for p in so_paths:
        if os.path.exists(p):
            so_path = p
    assert so_path, f"can't find coincurve lib, look in {package_path} and then modify this hook" 
    datas.append((so_path, 'coincurve'))

