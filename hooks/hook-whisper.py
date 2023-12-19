from PyInstaller.utils.hooks import collect_data_files, get_package_paths
import os

# Get the package path
package_path = get_package_paths('nvidia')[0]

# Collect data files
datas = collect_data_files('nvidia')

# Append the additional .dll or .so file
if os.name == 'nt':  # Windows
    # need to check because i dont have a gpu with windows to test
    dll_path = os.path.join(package_path, 'nvidia', 'cudnn', 'lib', 'libcudnn_ops_infer.dll')
    datas.append((dll_path, '.'))
    dll_path = os.path.join(package_path, 'nvidia', 'cudnn', 'lib', 'libcudnn_cnn_infer.dll')
    datas.append((dll_path, '.'))
elif os.name == 'posix':  # Linux/Mac
    so_path = os.path.join(package_path, 'nvidia', 'cudnn', 'lib', 'libcudnn_ops_infer.so.8')
    datas.append((so_path, '.'))
    so_path = os.path.join(package_path, 'nvidia', 'cudnn', 'lib', 'libcudnn_cnn_infer.so.8')
    datas.append((so_path, '.'))


package_path = get_package_paths('whisper')[0]
assets = os.path.join(package_path, 'assets')
datas.append((assets, './whisper/assets'))
    