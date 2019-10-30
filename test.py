# use this script to access the files in thanos, and combine the features from each instance
import pysftp

#connet to thanos to access the files in remote server
myHostName = "ssh.ece.ubc.ca"
myUserName = "angeli"
myPassword = "baileyD0g!"

with pysftp.Connection(host = myHostName, username = myUserName, password = myPassword) as sftp:
    print("Connection sucessfully stablished")
    myHostName = "thanos.ece.ubc.ca"
    myUserName = "angeli"
    myPassword = "baileyD0g!"

    with pysftp.Connection(host=myHostName, username=myUserName, password=myPassword) as sftp2:
        print("Connection 2 sucessfully stablished")