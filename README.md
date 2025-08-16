# Run Agentless on NoCode-bench

---

> NOTE: We modified the prompt in `agentless/fl/FL.py` and `agentless/repair/repair.py`. The remained code is original from official repository of [Agentless](https://github.com/OpenAutoCoder/Agentless).

## Environment Setup

```shell
conda create -n agentless python=3.11
pip install -r requirements.txt
```

## How to run?
### Preparation

You should generate the reporistory structure by running the following command:
```shell
python get_full_structure.py # For NoCode-bench Full
python get_verified_structure.py # For NoCode-bench Verified
```

After that, you should export the following environment variables in `exp.sh` at line 2:
```shell
export PROJECT_FILE_LOC=<path to your repo structures>
export OPENAI_API_KEY=<your api key>
export OPENAI_API_BASE=<your api base url>
```

To reproduce results, you can run the following command to reproduce the full NoCode-bench experiments:
```shell
bash exp.sh
```
