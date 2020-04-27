# Real-time Map visualization for COVID-19 china Data using python.
#### Introduction 
Using the Data provided by CSSEGISandData on this link : https://github.com/CSSEGISandData/COVID-19
China data is preprocessed in Data_preprocessing.py file. to get the needed
Data into data.json. Then Kafka Aphace is used to update the map live.
The map in index.js is built using Leaflet framework which you can learn more
about here https://leafletjs.com/reference-1.6.0.html
busdata1.py is the script that we are using to send data from our Apache kafka producer
to our consumer so we can update the map real-time.

You will need to install the following:
Java :  https://www.java.com/en/download/
open CMD and type 

java - version 
if you see java version it is installed correctly. 
After you  May need to add java to your variables
Go to windows start
Enter Edit Environment variables > Advanced > Environment Variables  
Click on Path variable  and add java path 

Next you will need to install Apache Kafka " Binary version IMPORTANT !"
You can find it here  https://kafka.apache.org/downloads.html

Also there is a binary version incleded in this respo under kafka folder.

Next you will need to add Kafka to your variables same steps
you will need to add kafka/bin/windows to your path. 

After adding Kafka to your path you should be able to use it from any where
on your machine. open the command prompt and type and type:
 
 kafka-topics.bat
 
 it will give you options means you downloaded kafka the right way.
 
 If you are having any problems. Please make sure you installed the binary version
 
 Two more steps:
 1) You will need to add three new folders in the kafka downloaded version directory
 firs folder goes into main kafka directory kafka/ 
 create data folder.
 inside kafka/data 
 create two more folders
 kafka
 zookeeper
 
 copy the path for your kafka/data/kafka
 copy the path for your kafka/data/zookeeper
 go to kafka/config/
 
 find zookeeper.properties 
 Open it with a text editor and edit the path:
 to your kafka/data/zookeeper mine looked like this:
 dataDir=D:/Projects/COVID-19_Visulize/kafka/data/zookeeper
 Second we need to edit server.properties:
 go to kafka/config/
 open server.properties and edit the path to your kafka/data/kafka
 log.dirs=D:/Projects/COVID-19_Visulize/kafka/data/kafka
 So all the logs can be stored in case you ran into an issue. 
 
 next you will need to install kafka.py
 
 pip install kafka  More intformation here:
 
 https://kafka-python.readthedocs.io/en/master/usage.html
 
 ## To run the visualization :
 
 Download this respo. 
 After following all of the steps above: 
 You will need to open 4 command prompt windows. 
 all command prompt should be in this path kafka/bin/windows

1) First command prompt run zookeeper :
zookeeper-server-start ../../config/zookeeper.properties

2) Second command prompt run server:

kafka-server-start.bat ../../config/server.properties

If you followed the guide above the right way you should have both
the zookeeper and the server running.

3) Create new topic: 

kafka-topics.bat --zookeeper 0.0.0.0:2181 --topic COVID-19-25  --create -partitions 1 --replication-factor 1

4) Run producer 

kafka-console-consumer --bootstrap-server localhost:9092 --topic COVID-19-25  --from-beginning
 
5) Run consumer 
kafka-console-consumer --bootstrap-server localhost:9092 --topic COVID-19-25  --from-beginning

6) Run flask server from the command prompt : Note: here you should be in the project directory 
python app.py

7) Run python script from the command prompt : Note: here you should be in the project directory 

Python dataline.py

You should see the map updating. 