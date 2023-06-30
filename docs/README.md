To develop locally, start liveserver and forward the port to local browser

To generate the necessary files:
```
sphinx-apidoc -o source/api_docs ../lab4d/ -f --templatedir template/
python source/obj2glb.py
```

To rebuild webpage:
```make clean; make html; mv build/html build/lab4d```