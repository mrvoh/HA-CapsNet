FROM python:3.6-slim

ADD https://github.com/openfaas-incubator/of-watchdog/releases/download/0.4.6/of-watchdog /usr/bin
RUN chmod +x /usr/bin/of-watchdog

RUN apt-get update -y && apt-get install -y python3-pip curl

WORKDIR /root/

RUN mkdir -p src
COPY src/requirements.txt .

RUN pip3 install --upgrade pip
RUN pip3 install urllib3==1.22
RUN pip3 install numpy==1.14.5
RUN pip3 install -r requirements.txt

ADD src .
RUN touch ./src/__init__.py

ENV fprocess="python3 index.py"

# Set timeouts to ensure function can complete
ENV write_timeout=600s
ENV read_timeout=600s
ENV exec_timeout=600s
ENV write_debug=true

ENV cgi_headers="true"
ENV mode="http"
ENV upstream_url="http://127.0.0.1:5000"

HEALTHCHECK --interval=5s CMD [ -e /tmp/.lock ] || exit 1

CMD ["of-watchdog"]