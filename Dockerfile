FROM python:3.7
WORKDIR /app
COPY . /app
ADD ./requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt
ADD . /app
EXPOSE 5000           
ENV FLASK_APP=app.py                                                            
ENTRYPOINT ["flask"]
CMD ["run","--host","0.0.0.0"]
