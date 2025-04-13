install conda env, activate, and

```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
cmake --build . --config Release

./grad_demo <PATH_TO_SCRIPTED_MODEL>
```
