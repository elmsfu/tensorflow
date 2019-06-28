# Build and execute tests

```shell
git submodule update --init
virtualenv -p `which python3` venv
. venv/bin/activate

(cd third_party/flatbuffers/python && python3 setup.py install)
pip install numpy

make -j$(nproc) tests
```

# Run larger graph

```shell
make cifar100-6cat-bw_out.bin cifar100-6cat-bw_tflite.bin
```
