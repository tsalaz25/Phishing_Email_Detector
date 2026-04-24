# Cyber_Project

## Run Loader uning Virtual Environment
In Mac Terminal
>python -m venv .venv
source .venv/bin/activate

Should See "(.venv) NAME@Mac Cyber_Project"
>pip install --upgrade pip
pip install datasets pandas pyarrow scikit-learn

Once PIP installs in VM
>python data_loader.py

use 'deactivate' to exit Virtual Environment
Afer Running, File is put into "ROOT->data->processed->phishing_emails.parquet"
- ".parquet" is a quicker, compressed .csv (thanks CS464)
- Virtual Environment is for usage of "pandas" python package