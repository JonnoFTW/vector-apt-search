# Vector Apt Search

Natural text search for apt. Run like this:

```shell
./vector_apt_search/vapt-search.py -k 5 "a package for "
```

| Argument | Description                                |
|----------|--------------------------------------------|
| `-k`     | Number of results to return. Defaults to 5 |
| `-v`     | Display the full package description       |
| `-r`     | Refresh cache                              |
| QUERY    | Query to search for                        |



## Requirements:

Install apt packages: `libapt-pkg-dev nvidia-cudnn nvidia-cuda-toolkit`

Install requirements.txt:

```shell
pip install -r requirements.txt
```


