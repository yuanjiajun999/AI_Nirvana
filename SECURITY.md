

## Security Check - 2024-07-24
Results of the latest security check:
+==============================================================================+

                               /$$$$$$            /$$
                              /$$__  $$          | $$
           /$$$$$$$  /$$$$$$ | $$  \__//$$$$$$  /$$$$$$   /$$   /$$
          /$$_____/ |____  $$| $$$$   /$$__  $$|_  $$_/  | $$  | $$
         |  $$$$$$   /$$$$$$$| $$_/  | $$$$$$$$  | $$    | $$  | $$
          \____  $$ /$$__  $$| $$    | $$_____/  | $$ /$$| $$  | $$
          /$$$$$$$/|  $$$$$$$| $$    |  $$$$$$$  |  $$$$/|  $$$$$$$
         |_______/  \_______/|__/     \_______/   \___/   \____  $$
                                                          /$$  | $$
                                                         |  $$$$$$/
  by safetycli.com                                        \______/

+==============================================================================+

 REPORT 

  Safety is using PyUp's free open-source vulnerability database. This
data is 30 days old and limited. 
  For real-time enhanced vulnerability data, fix recommendations, severity
reporting, cybersecurity support, team and project policy management and more
sign up at https://pyup.io or email sales@pyup.io

  Safety v3.2.4 is scanning for Vulnerabilities...
  Scanning dependencies in your stdin:

  absl-py, aiohttp, aiosignal, altair, annotated-types, anyio, argon2-cffi,
  argon2-cffi-bindings, arrow, astor, asttokens, astunparse, async-lru, async-
  timeout, attrs, authlib, babel, base58, beautifulsoup4, black, bleach,
  blinker, blis, boolean-py, cachecontrol, cachetools, catalogue, certifi, cffi,
  chardet, charset-normalizer, chex, click, cloudpathlib, cloudpickle, colorama,
  comm, confection, contourpy, cryptography, cssselect, cssutils, cycler,
  cyclonedx-python-lib, cymem, dataclasses-json, debugpy, decorator, defusedxml,
  differential-privacy, distro, dm-tree, dnspython, dparse, email-validator,
  entrypoints, etils, exceptiongroup, executing, fastapi, fastapi-cli,
  fastjsonschema, featuretools, filelock, flask, flatbuffers, flax, fonttools,
  fqdn, frozenlist, fsspec, gast, gensim, git-filter-repo, gitdb, gitpython,
  google-auth, google-auth-oauthlib, google-pasta, greenlet, grpcio, h11, h5py,
  holidays, html5lib, httpcore, httptools, httpx, huggingface-hub, idna,
  importlib-metadata, importlib-resources, iniconfig, intel-openmp, ipykernel,
  ipython, ipython-genutils, ipywidgets, isoduration, isort, itsdangerous, jax,
  jaxlib, jedi, jinja2, joblib, json5, jsonpatch, jsonpointer, jsonschema,
  jsonschema-specifications, jupyter, jupyter-console, jupyter-events, jupyter-
  lsp, jupyter-packaging, jupyter-client, jupyter-core, jupyter-server, jupyter-
  server-terminals, jupyterlab, jupyterlab-pygments, jupyterlab-server,
  jupyterlab-widgets, keras, kiwisolver, langchain, langchain-community,
  langchain-core, langchain-openai, langchain-text-splitters, langcodes,
  langsmith, language-data, libclang, license-expression, llvmlite, lxml,
  marisa-trie, markdown, markdown-it-py, markupsafe, marshmallow, matplotlib,
  matplotlib-inline, mdurl, mistune, mkl, ml-dtypes, more-itertools, mpmath,
  msgpack, multidict, murmurhash, mypy-extensions, namex, nbclassic, nbclient,
  nbconvert, nbformat, nest-asyncio, networkx, nltk, notebook, notebook-shim,
  numba, numpy, oauthlib, openai, opencv-python, opt-einsum, optax, optree,
  orbax-checkpoint, orjson, overrides, packageurl-python, packaging, pandas,
  pandocfilters, parso, pathspec, pillow, pip-api, pip-requirements-parser, pip-
  audit, platformdirs, pluggy, premailer, preshed, prometheus-client, prompt-
  toolkit, protobuf, psutil, pure-eval, py-serializable, pyarrow, pyasn1,
  pyasn1-modules, pycparser, pydantic, pydantic-core, pydeck, pygments,
  pyparsing, pytest, python-dateutil, python-dotenv, python-json-logger, python-
  multipart, pytz, pywin32, pywinpty, pyyaml, pyzmq, qtconsole, qtpy,
  referencing, regex, requests, requests-oauthlib, rfc3339-validator,
  rfc3986-validator, rich, rpds-py, rsa, ruamel-yaml, ruamel-yaml-clib,
  safetensors, safety, safety-schemas, scikit-learn, scipy, seaborn, send2trash,
  shap, shellingham, six, slicer, smart-open, smmap, sniffio, sortedcontainers,
  soupsieve, spacy, spacy-legacy, spacy-loggers, speechrecognition, sqlalchemy,
  srsly, stack-data, starlette, streamlit, sympy, tbb, tenacity, tensorboard,
  tensorboard-data-server, tensorflow, tensorflow-estimator, tensorflow-intel,
  tensorflow-io-gcs-filesystem, tensorstore, termcolor, terminado, thinc,
  threadpoolctl, tiktoken, tinycss2, tokenizers, toml, tomli, toolz, torch,
  torchvision, tornado, tqdm, traitlets, transformers, typer, types-python-
  dateutil, typing-inspect, typing-extensions, tzdata, tzlocal, ujson, uri-
  template, urllib3, uvicorn, validators, wasabi, watchdog, watchfiles, wcwidth,
  weasel, webcolors, webencodings, websocket-client, websockets, werkzeug,
  widgetsnbextension, woodwork, wrapt, yagmail, yarl, zipp

  Using the account yuanjiajun999@gmail.com and the Safety
  Commercial database
  Found and scanned 306 packages
  Timestamp 2024-07-24 13:17:27
  4 vulnerabilities reported
  0 vulnerabilities ignored
  4 remediations recommended

+==============================================================================+
 VULNERABILITIES REPORTED 
+==============================================================================+

-> Vulnerability found in torch version 2.3.1
   Vulnerability ID: 71670
   Affected spec: >=0
   ADVISORY: A vulnerability in the PyTorch's torch.distributed.rpc
   framework, specifically in versions prior to 2.2.2, allows for remote...
   Fixed versions: No known fix
   CVE-2024-5480
   For more information about this vulnerability, visit
   https://data.safetycli.com/v/71670/eda
   To ignore this vulnerability, use PyUp vulnerability id 71670 in safety’s
   ignore command-line argument or add the ignore to your safety policy file.


-> Vulnerability found in nltk version 3.8.1
   Vulnerability ID: 72089
   Affected spec: >=0
   ADVISORY: NLTK affected versions allow remote code execution if
   untrusted packages have pickled Python code, and the integrated data...
   Fixed versions: No known fix
   CVE-2024-39705
   For more information about this vulnerability, visit
   https://data.safetycli.com/v/72089/eda
   To ignore this vulnerability, use PyUp vulnerability id 72089 in safety’s
   ignore command-line argument or add the ignore to your safety policy file.


-> Vulnerability found in langchain version 0.2.9
   Vulnerability ID: 71924
   Affected spec: >=0
   ADVISORY: A Server-Side Request Forgery (SSRF) vulnerability
   exists in the Web Research Retriever component of langchain-...
   Fixed versions: No known fix
   CVE-2024-3095
   For more information about this vulnerability, visit
   https://data.safetycli.com/v/71924/eda
   To ignore this vulnerability, use PyUp vulnerability id 71924 in safety’s
   ignore command-line argument or add the ignore to your safety policy file.


-> Vulnerability found in jinja2 version 3.1.4
   Vulnerability ID: 70612
   Affected spec: >=0
   ADVISORY: In Jinja2, the from_string function is prone to Server
   Side Template Injection (SSTI) where it takes the "source" parameter as...
   Fixed versions: No known fix
   CVE-2019-8341 is CRITICAL SEVERITY => CVSS v3, BASE
   SCORE 9.8
   For more information about this vulnerability, visit
   https://data.safetycli.com/v/70612/eda
   To ignore this vulnerability, use PyUp vulnerability id 70612 in safety’s
   ignore command-line argument or add the ignore to your safety policy file.


+==============================================================================+
   REMEDIATIONS

-> torch version 2.3.1 was found, which has 1 vulnerability
                                                                              
   There is no known fix for this vulnerability. 
                                                                              
   For more information about the torch package and update options, visit
   https://data.safetycli.com/p/pypi/torch/eda/?from=2.3.1
   Always check for breaking changes when updating packages.
                                                                              
-> nltk version 3.8.1 was found, which has 1 vulnerability
                                                                              
   There is no known fix for this vulnerability. 
                                                                              
   For more information about the nltk package and update options, visit
   https://data.safetycli.com/p/pypi/nltk/eda/?from=3.8.1
   Always check for breaking changes when updating packages.
                                                                              
-> langchain version 0.2.9 was found, which has 1 vulnerability
                                                                              
   There is no known fix for this vulnerability. 
                                                                              
   For more information about the langchain package and update options, visit
   https://data.safetycli.com/p/pypi/langchain/eda/?from=0.2.9
   Always check for breaking changes when updating packages.
                                                                              
-> jinja2 version 3.1.4 was found, which has 1 vulnerability
                                                                              
   There is no known fix for this vulnerability. 
                                                                              
   For more information about the jinja2 package and update options, visit
   https://data.safetycli.com/p/pypi/jinja2/eda/?from=3.1.4
   Always check for breaking changes when updating packages.
                                                                              
+==============================================================================+

 Scan was completed. 4 vulnerabilities were reported. 4 remediations were 
 recommended. 

+==============================================================================+

  Safety is using PyUp's free open-source vulnerability database. This
data is 30 days old and limited. 
  For real-time enhanced vulnerability data, fix recommendations, severity
reporting, cybersecurity support, team and project policy management and more
sign up at https://pyup.io or email sales@pyup.io

+==============================================================================+
