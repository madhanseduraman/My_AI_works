# -*- coding: utf-8 -*-
"""
Created on Tue May 28 12:16:47 2019

@author: Vishnu.Kumar1
"""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
import os
from email.mime.base import MIMEBase

path = os.getcwd()

def email(recipient, name, intime, date):
    username = "GDSSolutionsLab@outlook.com"
    password = "ey@123456"
    msg = MIMEMultipart()
    msg['From'] = username
    msg['To'] = recipient
    msg['Subject'] = "Face Recognition POC Test Message"
    msg.attach(MIMEText("Hi " + str(name) + "," + "\n\nYour face has been identified and your attendance has been marked.\nYour in-time was recorded as " +  str(intime) + " on " + str(date) + "." + "\nIf it was not you, Please revert back to this mail.\n\nRegards,\nGDS Automation Solutions Lab"))
    try:
        mailServer = smtplib.SMTP('smtp-mail.outlook.com', 587)
        mailServer.ehlo()
        mailServer.starttls()
        mailServer.ehlo()
        mailServer.login(username, password)
        mailServer.sendmail(username, recipient, msg.as_string())
        mailServer.close()
    except Exception as e:
        print(e)
        pass
    
def FileEmail():
    username = "GDSSolutionsLab@outlook.com"
    password = "ey@123456"
    msg = MIMEMultipart()
    msg['From'] = username
    msg['Subject'] = 'Face Recognition POC Test Attendence Record'
    recipient = "Madhan.S@gds.ey.com"
    msg['To'] = recipient
    msg.attach(MIMEText("Hi Madhan,\n\nThe list of participants who attended today's POC is attached with this email.\n\nPFA\n\nRegards,\nGDS Automation Solutions Lab"))
    csv_file = 'Attendance.csv'
    
    with open(csv_file) as fp:
        record = MIMEBase('application', 'octet-stream')
        record.set_payload(fp.read())
        encoders.encode_base64(record)
        record.add_header('Content-Disposition', 'attachment',
                          filename=csv_file)
    msg.attach(record)
    
    try:
        server = smtplib.SMTP('smtp-mail.outlook.com', 587)
        server.ehlo()
        server.starttls()
        server.login(username, password)
        server.sendmail(username, recipient, msg.as_string())
        server.quit()
    except Exception as e:
        print(e)
        pass

def Template(image, name, recipient, gpn, intime):
    username = "GDSSolutionsLab@outlook.com"
    password = "ey@123456"
    msg = MIMEMultipart()
    msg['From'] = username
    msg['To'] = recipient
    msg['Subject'] = "Face Recognition Test Template"
    html = """\
    <!DOCTYPE html>
    <body>
        
        <div style="position:absolute; left:450px; top:100px;">
                <h2>WELCOME TO ELT – August 2019</h2></div>
    <!-- <div style="position:absolute; right:80px; top:80px;"> -->
    <div style="text-align: center; width:100px;position:absolute; right:80px; top:200px;height:100px;border:1px">
        <br><br>
    
        <img src={} alt="Italian Trulli">
    </div>
    <!-- </div> -->
    <div style="position:absolute; left:60px; top:5px;" > <img src="C:/Users/vishnu.kumar1/Desktop/Picture1.png" alt="Italian Trulli"></div>
		
    <div style="position:absolute; left:250px; top:250px;">
            <label style=" display: inline-block;width: 140px;text-align: right;">NAME    : </label>
        
            <input type="text" style="border: 0px none;"  placeholder="NAME" class="form-control" name="textwrite" value={} required="required" >
            <br><br>
            <label style=" display: inline-block;width: 140px;text-align: right;">E-Mail  :</label>
         
            <input type="text"  placeholder="E-Mail" class="form-control" name="textwrite" style="width:500px;" value={} required="required" style="border: 0px none;">
            <br><br>
            <label style=" display: inline-block;width: 200px;text-align: right;">GPN     :  </label>
        
            <input type="text"  placeholder="GPN" class="form-control" name="textwrite" value={} required="required" style="border: 0px none;" >
            <br><br>
            <label style=" display: inline-block;width: 140px;text-align: right;">In Time :  </label>
        
            <input type="text"  placeholder="In Time" class="form-control" name="textwrite" value={} required="required" style="border: 0px none;">
        
    </div>

    <div style=" position:absolute; left:500px; top:470px;">
            <h4>GDS Automation Solutions Lab</h2></div>
    </body>
    <html>
    """
    html = html.format(str(image), str(name), str(recipient), 
                       str(gpn), str(intime))
    part = MIMEText(html, 'html')
    msg.attach(part)
    
    try:
        mailServer = smtplib.SMTP('smtp-mail.outlook.com', 587)
        mailServer.ehlo()
        mailServer.starttls()
        mailServer.ehlo()
        mailServer.login(username, password)
        mailServer.sendmail(username, recipient, msg.as_string())
        mailServer.close()
    except Exception as e:
        print(e)
        pass