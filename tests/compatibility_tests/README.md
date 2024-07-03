For the purposes of testing compatibility between DPK, instructlab-sdg, fms-dgt

```bash
cd ./tests/compatibility_tests
git clone git@github.com:instructlab/sdg.git
pip install ./sdg
export PYTHONPATH="$PYTHONPATH:${PWD}"

git clone git@github.com:IBM/data-prep-kit.git
cd data-prep-kit
pip install ./data-prep-kit
```
