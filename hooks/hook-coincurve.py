from PyInstaller.utils.hooks import collect_data_files, get_package_paths
import os

# Get the package path
package_path = get_package_paths('coincurve')[0]

# Collect data files
datas = collect_data_files('coincurve')

# Append the additional .dll or .so file
if os.name == 'nt':  # Windows
    dll_path = os.path.join(package_path, 'coincurve', 'libsecp256k1.dll')
    datas.append((dll_path, 'coincurve'))
elif os.name == 'posix':  # Linux/Mac
    so_path = os.path.join(package_path, 'coincurve', 'libsecp256k1.so')
    datas.append((so_path, 'coincurve'))

