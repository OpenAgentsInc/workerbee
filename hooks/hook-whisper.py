from PyInstaller.utils.hooks import collect_data_files, get_package_paths
import os

# Get the package path
package_path = get_package_paths('nvidia')[0]

# Collect data files
datas = collect_data_files('nvidia')

# Append the additional .dll or .so file
if os.name == 'nt':  # Windows
    so_path = os.path.join(package_path, 'cudnn', 'lib', 'libcudnn_ops_infer.dll')
    datas.append((dll_path, '.'))
    so_path = os.path.join(package_path, 'cudnn', 'lib', 'libcudnn_cnn_infer.dll')
    datas.append((dll_path, '.'))
elif os.name == 'posix':  # Linux/Mac
    so_path = os.path.join(package_path, 'cudnn', 'lib', 'libcudnn_ops_infer.so.8')
    datas.append((so_path, '.'))
    so_path = os.path.join(package_path, 'cudnn', 'lib', 'libcudnn_cnn_infer.so.8')
    datas.append((so_path, '.'))
