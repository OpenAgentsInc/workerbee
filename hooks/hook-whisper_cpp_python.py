from PyInstaller.utils.hooks import collect_data_files, get_package_paths
import os

# Get the package path
package_path = get_package_paths('whisper_cpp_python')[0]

# Collect data files
datas = collect_data_files('whisper_cpp_python')

# Append the additional .dll or .so file
if os.name == 'nt':  # Windows
    dll_path = os.path.join(package_path, 'whisper_cpp_python', 'whisper.dll')
    datas.append((dll_path, 'whisper_cpp_python'))
elif os.name == 'posix':  # Linux/Mac
    so_path = os.path.join(package_path, 'whisper_cpp_python', 'libwhisper.so')
    datas.append((so_path, 'whisper_cpp_python'))
