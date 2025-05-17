FROM python:3.11

WORKDIR /app

RUN pip3 install lightgbm==3.2.1 scikit-learn pandas numpy lightgbmmodeloptimizer==0.0.6

ADD tests tests

CMD ['bash','-c','cd tests && python3 binary/binary_test.py' ]


